"""Scalar expressions and memory addressing shared by the pure emitters."""

from math import prod
from typing import assert_never

from .graph import Node, Operation, Shape, Symbol, UnsupportedGraphError, Value


def memref(value: Value) -> str:
    """Dense values keep the flat form; a strided view addresses through its own layout."""
    if value.is_dense:
        memory_type = f'!cute.memref<f32, gmem, align<4>, "({prod(value.shape)}):(1)">'
        return memory_type
    shape = ",".join(str(extent) for extent in value.shape)
    strides = ",".join(str(stride) for stride in value.strides)
    memory_type = f'!cute.memref<f32, gmem, align<4>, "({shape}):({strides})">'
    return memory_type


def logical_coordinate(shape: Shape, index: int) -> list[str]:
    """Split the flat right-major output index into one index per axis."""
    lines = [f"%rem0 = arith.addi %index{index}, %zero : i32"]
    axes = len(shape)
    for k in range(axes - 1, -1, -1):
        lines.append(f"%extent{k} = arith.constant {shape[k]} : i32")
        if k > 0:
            lines.append(f"%i{k} = arith.remsi %rem{axes - 1 - k}, %extent{k} : i32")
            lines.append(f"%rem{axes - k} = arith.divsi %rem{axes - 1 - k}, %extent{k} : i32")
        else:
            lines.append(f"%i0 = arith.addi %rem{axes - 1}, %zero : i32")
    args = ", ".join(f"%i{k}" for k in range(axes))
    types = ", ".join("i32" for _ in shape)
    marks = ",".join("?" for _ in shape)
    lines.append(f'%logical = cute.make_coord({args}) : ({types}) -> !cute.coord<"({marks})">')
    return lines


def expression(node: Node, names: dict[Symbol, str]) -> str:
    args = [names[name] for name in node.inputs]
    match node.operation:
        case Operation.BROADCAST:
            result = args[0]
        case Operation.SQUARE:
            result = f"arith.mulf {args[0]}, {args[0]} : f32"
        case Operation.NEGATIVE:
            result = f"arith.negf {args[0]} : f32"
        case Operation.RSQRT:
            result = f"math.rsqrt {args[0]} : f32"
        case Operation.ADD:
            result = f"arith.addf {args[0]}, {args[1]} : f32"
        case Operation.SUBTRACT:
            result = f"arith.subf {args[0]}, {args[1]} : f32"
        case Operation.MULTIPLY:
            result = f"arith.mulf {args[0]}, {args[1]} : f32"
        case Operation.REDUCE_SUM | Operation.TRANSPOSE:
            raise UnsupportedGraphError(f"{node.operation} has no scalar expression")
        case _:
            assert_never(node.operation)
    return result
