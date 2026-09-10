"""Decode the native export ABI into immutable tensor and native operations."""

from dataclasses import dataclass
from typing import assert_never

from mlx import core

from .events import Descriptor, ExportEvent, PrimitiveEvent
from .graph import (
    ArrayFunction,
    Graph,
    Node,
    Operation,
    Profile,
    Scalar,
    Strides,
    Symbol,
    UnsupportedGraphError,
    Value,
)
from .native import Collective, CollectiveOperation, GroupIndex, Matmul


@dataclass(frozen=True, kw_only=True)
class TensorOperation:
    node: Node
    stream: core.Stream

    @property
    def inputs(self) -> tuple[Symbol, ...]:
        inputs = self.node.inputs
        return inputs

    @property
    def output(self) -> Value:
        output = self.node.output
        return output


@dataclass(frozen=True, kw_only=True)
class Captured:
    signature: Graph
    operations: tuple[TensorOperation | Matmul | Collective, ...]

    def tensor_graph(self) -> Graph:
        nodes: list[Node] = []
        for operation in self.operations:
            if not isinstance(operation, TensorOperation):
                raise UnsupportedGraphError(f"expected a tensor region, got {operation.operation}")
            nodes.append(operation.node)
        graph = Graph(
            inputs=self.signature.inputs,
            constants=self.signature.constants,
            nodes=tuple(nodes),
            outputs=self.signature.outputs,
        )
        return graph


def descriptor(raw: Descriptor, *, strides: Strides | None = None) -> Value:
    name, shape, dtype = raw
    if dtype != core.float32:
        raise UnsupportedGraphError(f"expected float32, got {dtype} at {name}")
    if strides is None:
        result = Value.dense(name=Symbol(name), shape=tuple(shape))
        return result
    value = Value(name=Symbol(name), shape=tuple(shape), strides=strides)
    return value


def trace(function: ArrayFunction, profiles: tuple[Profile, ...]) -> list[ExportEvent]:
    """Trace dense placeholders; live strides affect addressing, not graph identity."""
    events: list[ExportEvent] = []
    placeholders = [core.zeros(profile.shape, dtype=core.float32) for profile in profiles]
    core.export_function(events.append, function, *placeholders)
    return events


def from_events(events: list[ExportEvent], profiles: tuple[Profile, ...]) -> Captured:
    """Require one signature and validate its layout before decoding operations."""
    inputs: tuple[Value, ...] = ()
    outputs: tuple[Value, ...] = ()
    constants: list[Scalar] = []
    operations: list[TensorOperation | Matmul | Collective] = []
    seen: set[str] = set()
    for event in events:
        if event["type"] != "primitive":
            if event["type"] in seen:
                raise UnsupportedGraphError(f"duplicate export header: {event['type']}")
            seen.add(event["type"])
        if event["type"] == "inputs":
            inputs = input_values(event["inputs"], profiles)
        elif event["type"] == "outputs":
            outputs = tuple(descriptor(raw) for raw in event["outputs"])
        elif event["type"] == "constants":
            constants.extend(parse_constant(name, value) for name, value in event["constants"])
        elif event["type"] == "keyword_inputs":
            if event["keywords"]:
                raise UnsupportedGraphError("keyword array arguments are unsupported")
        elif event["type"] == "primitive":
            operations.append(parse_primitive(event))
        else:
            assert_never(event)
    if seen != {"inputs", "outputs", "constants", "keyword_inputs"}:
        raise UnsupportedGraphError(f"incomplete export signature: {sorted(seen)}")
    signature = Graph(inputs=inputs, constants=tuple(constants), nodes=(), outputs=outputs)
    captured = Captured(signature=signature, operations=tuple(operations))
    return captured


def parse_constant(name: str, value: core.array) -> Scalar:
    if value.ndim != 0 or value.dtype != core.float32:
        raise UnsupportedGraphError("captured constants must be float32 scalars")
    scalar = value.item()
    assert isinstance(scalar, float), "float32 scalar callback must return float"
    constant = Scalar(name=Symbol(name), value=scalar)
    return constant


def input_values(raw_inputs: list[Descriptor], profiles: tuple[Profile, ...]) -> tuple[Value, ...]:
    if len(raw_inputs) != len(profiles):
        raise UnsupportedGraphError(f"traced {len(raw_inputs)} inputs for {len(profiles)} profiles")
    values: list[Value] = []
    for raw, profile in zip(raw_inputs, profiles, strict=True):
        if tuple(raw[1]) != profile.shape:
            raise UnsupportedGraphError(f"profile shape {profile.shape} differs from {raw[1]}")
        values.append(descriptor(raw, strides=profile.strides))
    result = tuple(values)
    return result


def capture(function: ArrayFunction, profiles: tuple[Profile, ...]) -> Graph:
    captured = from_events(trace(function, profiles), profiles)
    graph = captured.tensor_graph()
    return graph


def parse_primitive(event: PrimitiveEvent) -> TensorOperation | Matmul | Collective:
    name, state = event["name"], event["arguments"]
    inputs = tuple(descriptor(raw).name for raw in event["inputs"])
    outputs = tuple(descriptor(raw) for raw in event["outputs"])
    if len(outputs) != 1:
        raise UnsupportedGraphError(f"unsupported output arity for {name}")
    output, stream = outputs[0], event["stream"]
    if name in ("AllReduce", "AllGather", "ReduceScatter"):
        if len(inputs) != 1:
            raise UnsupportedGraphError(f"unsupported input arity for {name}")
        if "group" not in event or "group_index" not in event:
            raise UnsupportedGraphError(f"missing communicator for {name}")
        index = event["group_index"]
        if type(index) is not int or index < 0:
            raise UnsupportedGraphError(f"invalid communicator index for {name}: {index}")
        result = Collective(
            operation=collective_operation(event),
            input=inputs[0],
            output=output,
            stream=stream,
            group=event["group"],
            group_index=GroupIndex(index),
        )
        return result
    if "group" in event or "group_index" in event:
        raise UnsupportedGraphError(f"unexpected communicator for {name}")
    if name == "Matmul":
        if len(inputs) != 2:
            raise UnsupportedGraphError("Matmul requires two inputs")
        if state:
            raise UnsupportedGraphError("Matmul must have no primitive arguments")
        result = Matmul(left=inputs[0], right=inputs[1], output=output, stream=stream)
        return result
    node = Node(operation=tensor_operation(event, output), inputs=inputs, output=output)
    placed = TensorOperation(node=node, stream=stream)
    return placed


def tensor_operation(event: PrimitiveEvent, output: Value) -> Operation:
    """Accept only the state represented by the tensor graph's operation and shape."""
    name, state = event["name"], event["arguments"]
    match name:
        case "Reduce":
            if len(event["inputs"]) != 1:
                raise UnsupportedGraphError("Reduce requires one input")
            if state != [2, [len(event["inputs"][0][1]) - 1]]:
                raise UnsupportedGraphError("only sum over the last axis is supported")
            operation = Operation.REDUCE_SUM
        case "Sqrt":
            if state != [True] or type(state[0]) is not bool:
                raise UnsupportedGraphError("only reciprocal square root is supported")
            operation = Operation.RSQRT
        case "Transpose":
            if state != [[1, 0]]:
                raise UnsupportedGraphError("only a two-dimensional transpose is supported")
            operation = Operation.TRANSPOSE
        case "Broadcast":
            if state != [output.shape]:
                raise UnsupportedGraphError("Broadcast arguments must match its output shape")
            operation = Operation.BROADCAST
        case "Add" | "Subtract" | "Multiply" | "Negative" | "Square":
            if state:
                raise UnsupportedGraphError(f"{name} must have no primitive arguments")
            operation = Operation(name)
        case _:
            raise UnsupportedGraphError(f"unsupported MLX primitive: {name}")
    return operation


def collective_operation(event: PrimitiveEvent) -> CollectiveOperation:
    """Decode C++ enum ordinals, which differ between Reduce and ReduceScatter."""
    if any(type(argument) is not int for argument in event["arguments"]):
        raise UnsupportedGraphError(
            f"collective arguments must be integer enum ordinals: {event['name']}"
        )
    match event["name"], event["arguments"]:
        case "AllReduce", [2]:
            result = CollectiveOperation.SUM
            return result
        case "AllReduce", [4]:
            result = CollectiveOperation.MIN
            return result
        case "AllReduce", [5]:
            result = CollectiveOperation.MAX
            return result
        case "AllGather", []:
            result = CollectiveOperation.GATHER
            return result
        case "ReduceScatter", [0]:
            result = CollectiveOperation.SUM_SCATTER
            return result
        case _:
            raise UnsupportedGraphError(
                f"unsupported collective: {event['name']} {event['arguments']}"
            )
