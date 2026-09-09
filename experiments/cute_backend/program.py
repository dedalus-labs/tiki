"""Lower a tensor program into CuTe regions and explicit native operations."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import mlx.core as mx

from graph import (
    ExportEvent,
    Graph,
    Profile,
    Shape,
    UnsupportedGraphError,
    Value,
    descriptor,
    from_events,
    parse_node,
)
from lowering import Lowered, Schedule, lower

NATIVE_OPERATIONS = frozenset(("Matmul", "AllReduce", "AllGather", "ReduceScatter"))


@dataclass(frozen=True)
class Native:
    operation: str
    inputs: tuple[str, ...]
    output: Value
    stream: mx.Stream
    function: Callable[..., mx.array]
    group_index: int | None


@dataclass(frozen=True)
class Kernel:
    lowered: Lowered
    stream: mx.Stream
    operation: str = "CuTe"


@dataclass(frozen=True)
class Program:
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    constants: tuple[tuple[str, float], ...]
    stages: tuple[Native | Kernel, ...]

    @property
    def output_shapes(self) -> tuple[Shape, ...]:
        return tuple(value.shape for value in self.outputs)

    def launch(
        self,
        inputs: tuple[mx.array, ...],
        launch_kernel: Callable[
            [Lowered, tuple[mx.array, ...], mx.Stream], tuple[mx.array, ...]
        ],
    ) -> tuple[mx.array, ...]:
        values = dict(zip((value.name for value in self.inputs), inputs, strict=True))
        values.update(
            (name, mx.array(value, dtype=mx.float32)) for name, value in self.constants
        )
        completions: dict[int, mx.array] = {}
        for stage in self.stages:
            if isinstance(stage, Kernel):
                region = stage.lowered.graph
                arguments = tuple(values[value.name] for value in region.inputs)
                outputs = launch_kernel(stage.lowered, arguments, stage.stream)
                values.update(
                    zip((value.name for value in region.outputs), outputs, strict=True)
                )
                continue
            arguments = tuple(values[name] for name in stage.inputs)
            if stage.group_index in completions:
                with mx.stream(stage.stream):
                    arguments = (
                        mx.depends(arguments[0], completions[stage.group_index]),
                    )
            output = stage.function(*arguments)
            values[stage.output.name] = output
            if stage.group_index is not None:
                completions[stage.group_index] = output
        return tuple(values[value.name] for value in self.outputs)


def native(event: ExportEvent) -> Native:
    operation, state, stream = event["name"], event["arguments"], event["stream"]
    inputs = tuple(descriptor(raw).name for raw in event["inputs"])
    outputs = tuple(descriptor(raw) for raw in event["outputs"])
    arity = 2 if operation == "Matmul" else 1
    if len(inputs) != arity or len(outputs) != 1:
        raise UnsupportedGraphError(f"unsupported arity for {operation}")
    if operation == "Matmul":
        if state:
            raise UnsupportedGraphError("Matmul must have no primitive arguments")
        return Native(
            operation,
            inputs,
            outputs[0],
            stream,
            partial(mx.matmul, stream=stream),
            None,
        )
    operations = {
        ("AllReduce", (2,)): mx.distributed.all_sum,
        ("AllReduce", (4,)): mx.distributed.all_min,
        ("AllReduce", (5,)): mx.distributed.all_max,
        ("AllGather", ()): mx.distributed.all_gather,
        ("ReduceScatter", (0,)): mx.distributed.sum_scatter,
    }
    key = (operation, tuple(state))
    if key not in operations:
        raise UnsupportedGraphError(f"unsupported collective: {operation} {state}")
    function = partial(operations[key], group=event["group"], stream=stream)
    return Native(operation, inputs, outputs[0], stream, function, event["group_index"])


def lower_region(
    events: list[ExportEvent],
    values: dict[str, Value],
    constants: tuple[tuple[str, float], ...],
    required: set[str],
    schedule: Schedule,
) -> Kernel:
    nodes = tuple(parse_node(event) for event in events)
    produced = {node.output.name for node in nodes}
    constant_names = {name for name, _ in constants}
    names = dict.fromkeys(
        name
        for node in nodes
        for name in node.inputs
        if name not in produced | constant_names
    )
    graph = Graph(
        tuple(values[name] for name in names),
        constants,
        nodes,
        tuple(node.output for node in nodes if node.output.name in required),
    )
    return Kernel(lower(graph, schedule), events[0]["stream"])


def partition(
    events: list[ExportEvent], profiles: tuple[Profile, ...], schedule: Schedule
) -> Program:
    headers = [event for event in events if event["type"] != "primitive"]
    signature = from_events(headers, profiles)
    primitives = [event for event in events if event["type"] == "primitive"]
    values = {value.name: value for value in signature.inputs}
    values.update((name, Value(name, ())) for name, _ in signature.constants)
    for event in primitives:
        values.update((raw[0], descriptor(raw)) for raw in event["outputs"])
    stages: list[Native | Kernel] = []
    position = 0
    while position < len(primitives):
        event = primitives[position]
        if event["name"] in NATIVE_OPERATIONS:
            stages.append(native(event))
            position += 1
            continue
        end = position
        shape = ()
        while end < len(primitives):
            candidate = primitives[end]
            if (
                candidate["name"] in NATIVE_OPERATIONS
                or candidate["stream"] != event["stream"]
            ):
                break
            node = parse_node(candidate)
            if shape and node.output.shape not in ((), shape):
                break
            shape = node.output.shape or shape
            end += 1
        required = {value.name for value in signature.outputs}
        required.update(raw[0] for later in primitives[end:] for raw in later["inputs"])
        stages.append(
            lower_region(
                primitives[position:end],
                values,
                signature.constants,
                required,
                schedule,
            )
        )
        position = end
    return Program(
        signature.inputs, signature.outputs, signature.constants, tuple(stages)
    )
