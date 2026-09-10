"""Immutable, float32 tensor graphs consumed by the pure MLIR emitters."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import prod
from typing import NamedTuple, NewType, assert_never

from mlx import core

type Shape = tuple[int, ...]
type Strides = tuple[int, ...]
type Arrays = tuple[core.array, ...]
type ArrayResult = core.array | Arrays
# Arity is determined by the traced graph; every argument and result is an array.
type ArrayFunction = Callable[..., ArrayResult]
Symbol = NewType("Symbol", str)


class UnsupportedGraphError(ValueError):
    """The exported graph is outside the compiler contract."""


def dense_strides(shape: Shape) -> Strides:
    """Compute right-major strides, including zero-sized and scalar layouts."""
    strides = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
    return strides


class Profile(NamedTuple):
    shape: Shape
    strides: Strides


class Scalar(NamedTuple):
    name: Symbol
    value: float


class Operation(StrEnum):
    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    NEGATIVE = "Negative"
    SQUARE = "Square"
    BROADCAST = "Broadcast"
    REDUCE_SUM = "ReduceSum"
    RSQRT = "Rsqrt"
    TRANSPOSE = "Transpose"

    @property
    def arity(self) -> int:
        match self:
            case Operation.ADD | Operation.SUBTRACT | Operation.MULTIPLY:
                return 2
            case (
                Operation.NEGATIVE
                | Operation.SQUARE
                | Operation.BROADCAST
                | Operation.REDUCE_SUM
                | Operation.RSQRT
                | Operation.TRANSPOSE
            ):
                return 1
        assert_never(self)


@dataclass(frozen=True, kw_only=True)
class Value:
    name: Symbol
    shape: Shape
    strides: Strides

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.strides):
            raise UnsupportedGraphError(f"shape/stride rank mismatch at {self.name}")

    @classmethod
    def dense(cls, name: Symbol, shape: Shape) -> Value:
        value = cls(name=name, shape=shape, strides=dense_strides(shape))
        return value

    @property
    def is_dense(self) -> bool:
        """Ignore strides of extent-one axes, which carry no layout information."""
        dense = dense_strides(self.shape)
        result = all(
            extent == 1 or stride == expected
            for extent, stride, expected in zip(self.shape, self.strides, dense, strict=True)
        )
        return result


@dataclass(frozen=True, kw_only=True)
class Node:
    operation: Operation
    inputs: tuple[Symbol, ...]
    output: Value

    def __post_init__(self) -> None:
        if len(self.inputs) != self.operation.arity:
            raise UnsupportedGraphError(f"unsupported arity for {self.operation}")


@dataclass(frozen=True, kw_only=True)
class Graph:
    inputs: tuple[Value, ...]
    constants: tuple[Scalar, ...]
    nodes: tuple[Node, ...]
    outputs: tuple[Value, ...]

    def __post_init__(self) -> None:
        if not self.outputs:
            raise UnsupportedGraphError("at least one array output is required")
        if any(prod(output.shape) >= 2**31 - 1024 for output in self.outputs):
            raise UnsupportedGraphError("element count exceeds signed 32-bit indexing")

    @property
    def output(self) -> Symbol:
        """Require one output for schedules that fuse a single result."""
        if len(self.outputs) != 1:
            raise UnsupportedGraphError("this schedule requires exactly one output")
        result = self.outputs[0].name
        return result

    @property
    def shape(self) -> Shape:
        shape = self.outputs[0].shape
        return shape
