"""Capture a supported MLX graph through its native export callback."""

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import TypedDict

import mlx.core as mx

Shape = tuple[int, ...]

Strides = tuple[int, ...]
Profile = tuple[Shape, Strides]


def dense_strides(shape: Shape) -> Strides:
    """Right-major strides, the layout of every MLX output and of a fresh array."""
    return tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))


Descriptor = tuple[str, Shape, mx.Dtype]
ArrayFunction = Callable[..., mx.array]


class UnsupportedGraphError(ValueError):
    """The exported graph is outside the elementwise compiler contract."""


class ExportEvent(TypedDict, total=False):
    type: str
    inputs: list[Descriptor]
    outputs: list[Descriptor]
    name: str
    constants: list[tuple[str, mx.array]]
    keywords: list[tuple[str, str]]
    arguments: list[bool | int | list[int] | tuple[int, ...]]


@dataclass(frozen=True)
class Value:
    name: str
    shape: Shape
    strides: Strides = ()

    def __post_init__(self) -> None:
        if self.strides == () and self.shape != ():
            object.__setattr__(self, "strides", dense_strides(self.shape))

    @property
    def is_dense(self) -> bool:
        """Right-major and contiguous; strides of extent-1 axes carry no information."""
        dense = dense_strides(self.shape)
        return all(
            extent == 1 or stride == expected
            for extent, stride, expected in zip(self.shape, self.strides, dense)
        )


@dataclass(frozen=True)
class Node:
    operation: str
    inputs: tuple[str, ...]
    output: Value


@dataclass(frozen=True)
class Graph:
    inputs: tuple[Value, ...]
    constants: tuple[tuple[str, float], ...]
    nodes: tuple[Node, ...]
    outputs: tuple[Value, ...]

    @property
    def output(self) -> str:
        """The single output's name; schedules that fuse one output ask for it."""
        if len(self.outputs) != 1:
            raise UnsupportedGraphError("this schedule requires exactly one output")
        return self.outputs[0].name

    @property
    def shape(self) -> Shape:
        return self.outputs[0].shape


def descriptor(raw: Descriptor, strides: Strides = ()) -> Value:
    name, shape, dtype = raw
    if dtype != mx.float32:
        raise UnsupportedGraphError(f"expected float32, got {dtype} at {name}")
    return Value(name, tuple(shape), strides)


def capture(function: ArrayFunction, profiles: tuple[Profile, ...]) -> Graph:
    """Trace ``function`` on dense placeholders; record each input's strides.

    Strides do not change the traced graph, only how the lowering addresses
    each input, so tracing stays on dense placeholders.
    """
    events: list[ExportEvent] = []
    placeholders = [mx.zeros(shape, dtype=mx.float32) for shape, _ in profiles]
    mx.export_function(events.append, function, *placeholders)
    headers = {event["type"]: event for event in events if event["type"] != "primitive"}
    raw_inputs = headers["inputs"]["inputs"]
    if len(raw_inputs) != len(profiles):
        raise UnsupportedGraphError(
            f"traced {len(raw_inputs)} inputs for {len(profiles)} profiles"
        )
    inputs = tuple(
        descriptor(raw, strides) for raw, (_, strides) in zip(raw_inputs, profiles)
    )
    outputs = tuple(descriptor(raw) for raw in headers["outputs"]["outputs"])
    if not outputs:
        raise UnsupportedGraphError("at least one array output is required")
    constants = []
    for name, value in headers["constants"]["constants"]:
        if value.ndim != 0 or value.dtype != mx.float32:
            raise UnsupportedGraphError("captured constants must be float32 scalars")
        constants.append((name, float(value.item())))
    nodes = tuple(parse_node(event) for event in events if event["type"] == "primitive")
    if any(prod(output.shape) >= 2**31 - 1024 for output in outputs):
        raise UnsupportedGraphError("element count exceeds signed 32-bit indexing")
    return Graph(inputs, tuple(constants), nodes, outputs)


def parse_node(event: ExportEvent) -> Node:
    operation = event["name"]
    if operation == "Reduce":
        axes = [len(event["inputs"][0][1]) - 1]
        if event["arguments"] != [2, axes]:
            raise UnsupportedGraphError("only sum over the last axis is supported")
        operation = "ReduceSum"
    if operation == "Sqrt":
        if event["arguments"] != [True]:
            raise UnsupportedGraphError("only reciprocal square root is supported")
        operation = "Rsqrt"
    if operation == "Transpose" and event["arguments"] != [[1, 0]]:
        raise UnsupportedGraphError("only a two-dimensional transpose is supported")
    arity = {
        "Add": 2,
        "Subtract": 2,
        "Multiply": 2,
        "Negative": 1,
        "Square": 1,
        "Broadcast": 1,
        "ReduceSum": 1,
        "Rsqrt": 1,
        "Transpose": 1,
    }
    if operation not in arity:
        raise UnsupportedGraphError(f"unsupported MLX primitive: {operation}")
    inputs = tuple(descriptor(raw) for raw in event["inputs"])
    outputs = tuple(descriptor(raw) for raw in event["outputs"])
    if len(inputs) != arity[operation] or len(outputs) != 1:
        raise UnsupportedGraphError(f"unsupported arity for {operation}")
    return Node(operation, tuple(value.name for value in inputs), outputs[0])
