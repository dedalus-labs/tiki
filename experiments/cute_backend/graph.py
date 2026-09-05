"""Capture a supported MLX graph through its native export callback."""

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import TypedDict

import mlx.core as mx

Shape = tuple[int, ...]
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


@dataclass(frozen=True)
class Value:
    name: str
    shape: Shape


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
    output: str
    shape: Shape


def descriptor(raw: Descriptor) -> Value:
    name, shape, dtype = raw
    if dtype != mx.float32:
        raise UnsupportedGraphError(f"expected float32, got {dtype} at {name}")
    return Value(name, tuple(shape))


def capture(function: ArrayFunction, shapes: tuple[Shape, ...]) -> Graph:
    events: list[ExportEvent] = []
    placeholders = [mx.zeros(shape, dtype=mx.float32) for shape in shapes]
    mx.export_function(events.append, function, *placeholders)
    headers = {event["type"]: event for event in events if event["type"] != "primitive"}
    inputs = tuple(descriptor(raw) for raw in headers["inputs"]["inputs"])
    outputs = headers["outputs"]["outputs"]
    if len(outputs) != 1:
        raise UnsupportedGraphError("exactly one array output is required")
    output = descriptor(outputs[0])
    if any(value.shape not in ((), output.shape) for value in inputs):
        raise UnsupportedGraphError("inputs must have the output shape or be scalars")
    constants = []
    for name, value in headers["constants"]["constants"]:
        if value.ndim != 0 or value.dtype != mx.float32:
            raise UnsupportedGraphError("captured constants must be float32 scalars")
        constants.append((name, float(value.item())))
    nodes = tuple(
        parse_node(event, output.shape)
        for event in events
        if event["type"] == "primitive"
    )
    if prod(output.shape) >= 2**31 - 1024:
        raise UnsupportedGraphError("element count exceeds signed 32-bit indexing")
    return Graph(inputs, tuple(constants), nodes, output.name, output.shape)


def parse_node(event: ExportEvent, shape: Shape) -> Node:
    operation = event["name"]
    arity = {
        "Add": 2,
        "Subtract": 2,
        "Multiply": 2,
        "Negative": 1,
        "Square": 1,
        "Broadcast": 1,
    }
    if operation not in arity:
        raise UnsupportedGraphError(f"unsupported MLX primitive: {operation}")
    inputs = tuple(descriptor(raw) for raw in event["inputs"])
    outputs = tuple(descriptor(raw) for raw in event["outputs"])
    if len(inputs) != arity[operation] or len(outputs) != 1:
        raise UnsupportedGraphError(f"unsupported arity for {operation}")
    if any(value.shape not in ((), shape) for value in (*inputs, *outputs)):
        raise UnsupportedGraphError(f"unsupported shape in {operation}")
    return Node(operation, tuple(value.name for value in inputs), outputs[0])
