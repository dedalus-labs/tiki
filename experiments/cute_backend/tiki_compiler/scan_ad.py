"""Array views and matrix-affine combines used by scan differentiation."""

from collections.abc import Callable
from functools import reduce
from operator import add

from mlx import core

from .graph import Arrays

type FlatCombine = Callable[..., Arrays]


def view(
    array: core.array, axis: int, start: int | None, stop: int | None, step: int = 1
) -> core.array:
    index: list[slice] = [slice(None)] * array.ndim
    index[axis] = slice(start, stop, step)
    view = array[tuple(index)]
    return view


def flip(array: core.array, axis: int) -> core.array:
    """A negatively strided view; the kernels consume it in place."""
    reversed_view = view(array, axis, None, None, -1)
    return reversed_view


def basis(array: core.array, value: float) -> core.array:
    """A constant tangent the graph capture accepts: a float32 scalar broadcast."""
    constant = core.broadcast_to(core.array(value, dtype=core.float32), array.shape)
    return constant


def affine_combine(size: int) -> FlatCombine:
    """``(A_l, b_l) o (A_r, b_r) = (A_r A_l, A_r b_l + b_r)`` over ``size*size + size`` leaves."""

    def combine(*args: core.array) -> tuple[core.array, ...]:
        a_left, b_left = args[: size * size], args[size * size : size * size + size]
        a_right, b_right = (
            args[size * size + size : 2 * size * size + size],
            args[2 * size * size + size :],
        )
        matrix = [
            reduce(
                add,
                (
                    a_right[row * size + inner] * a_left[inner * size + column]
                    for inner in range(size)
                ),
            )
            for row in range(size)
            for column in range(size)
        ]
        vector = [
            reduce(
                add,
                (a_right[row * size + inner] * b_left[inner] for inner in range(size)),
            )
            + b_right[row]
            for row in range(size)
        ]
        result = (*matrix, *vector)
        return result

    return combine


def matvec(size: int, transpose: bool) -> Callable[..., tuple[core.array, ...]]:
    """``M v`` (or ``M^T v``) over ``size*size`` matrix leaves and ``size`` vector leaves."""

    def function(*args: core.array) -> tuple[core.array, ...]:
        matrix, vector = args[: size * size], args[size * size :]
        if transpose:
            result = tuple(
                reduce(
                    add,
                    (matrix[row * size + column] * vector[row] for row in range(size)),
                )
                for column in range(size)
            )
            return result
        result = tuple(
            reduce(
                add,
                (matrix[row * size + column] * vector[column] for column in range(size)),
            )
            for row in range(size)
        )
        return result

    return function
