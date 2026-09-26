"""Sequential float64 derivative oracles for a two-dimensional complex recurrence.

The four leaves represent real/imaginary parts of coefficients and increments.
At position t, product[t] = coefficient[t] * product[t-1] and
state[t] = coefficient[t] * state[t-1] + increment[t]. Position zero is the input.
These loops share neither the GPU scan tree nor MLX's differentiation primitives.
"""

from collections.abc import Sequence

import numpy as np
from mlx import core
from numpy.typing import NDArray

type ComplexGrid = NDArray[np.complex128]
type RealGrid = NDArray[np.float64]


def complex_pair(leaves: Sequence[core.array]) -> tuple[ComplexGrid, ComplexGrid]:
    real, imaginary, increment_real, increment_imaginary = (
        np.asarray(leaf, dtype=np.float64) for leaf in leaves
    )
    values = (real + 1j * imaginary, increment_real + 1j * increment_imaginary)
    return values


def real_parts(first: ComplexGrid, second: ComplexGrid) -> tuple[RealGrid, ...]:
    leaves = (first.real, first.imag, second.real, second.imag)
    return leaves


def forward(coefficient: ComplexGrid, increment: ComplexGrid) -> tuple[ComplexGrid, ComplexGrid]:
    product, state = coefficient.copy(), increment.copy()
    for column in range(1, coefficient.shape[1]):
        product[:, column] = coefficient[:, column] * product[:, column - 1]
        state[:, column] = coefficient[:, column] * state[:, column - 1] + increment[:, column]
    outputs = (product, state)
    return outputs


def backward(
    coefficient: ComplexGrid,
    increment: ComplexGrid,
    product_grad: ComplexGrid,
    state_grad: ComplexGrid,
) -> tuple[ComplexGrid, ComplexGrid]:
    product, state = forward(coefficient, increment)
    coefficient_grad, increment_grad = np.zeros_like(coefficient), np.zeros_like(increment)
    for column in range(coefficient.shape[1] - 1, 0, -1):
        coefficient_grad[:, column] = (
            product_grad[:, column] * product[:, column - 1].conj()
            + state_grad[:, column] * state[:, column - 1].conj()
        )
        increment_grad[:, column] = state_grad[:, column]
        product_grad[:, column - 1] += product_grad[:, column] * coefficient[:, column].conj()
        state_grad[:, column - 1] += state_grad[:, column] * coefficient[:, column].conj()
    coefficient_grad[:, 0], increment_grad[:, 0] = product_grad[:, 0], state_grad[:, 0]
    result = (coefficient_grad, increment_grad)
    return result


def tangent(
    coefficient: ComplexGrid,
    increment: ComplexGrid,
    coefficient_tangent: ComplexGrid,
    increment_tangent: ComplexGrid,
) -> tuple[ComplexGrid, ComplexGrid]:
    product, state = forward(coefficient, increment)
    product_tangent, state_tangent = coefficient_tangent.copy(), increment_tangent.copy()
    for column in range(1, coefficient.shape[1]):
        product_tangent[:, column] = (
            coefficient_tangent[:, column] * product[:, column - 1]
            + coefficient[:, column] * product_tangent[:, column - 1]
        )
        state_tangent[:, column] = (
            coefficient_tangent[:, column] * state[:, column - 1]
            + coefficient[:, column] * state_tangent[:, column - 1]
            + increment_tangent[:, column]
        )
    result = (product_tangent, state_tangent)
    return result


def complex_vjp(
    inputs: Sequence[core.array], cotangents: Sequence[core.array]
) -> tuple[RealGrid, ...]:
    gradients = backward(*complex_pair(inputs), *complex_pair(cotangents))
    result = real_parts(*gradients)
    return result


def complex_jvp(
    inputs: Sequence[core.array], tangents: Sequence[core.array]
) -> tuple[RealGrid, ...]:
    derivatives = tangent(*complex_pair(inputs), *complex_pair(tangents))
    result = real_parts(*derivatives)
    return result


def affine_vjp(
    inputs: Sequence[core.array], cotangents: Sequence[core.array]
) -> tuple[RealGrid, ...]:
    coefficient, increment = (np.asarray(leaf, dtype=np.complex128) for leaf in inputs)
    product_grad, state_grad = (np.asarray(leaf, dtype=np.complex128) for leaf in cotangents)
    gradients = backward(coefficient, increment, product_grad, state_grad)
    result = tuple(value.real for value in gradients)
    return result


def affine_jvp(
    inputs: Sequence[core.array], tangents: Sequence[core.array]
) -> tuple[RealGrid, ...]:
    coefficient, increment = (np.asarray(leaf, dtype=np.complex128) for leaf in inputs)
    coefficient_tangent, increment_tangent = (
        np.asarray(leaf, dtype=np.complex128) for leaf in tangents
    )
    derivatives = tangent(coefficient, increment, coefficient_tangent, increment_tangent)
    result = tuple(value.real for value in derivatives)
    return result
