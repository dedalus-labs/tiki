# Copyright © 2026 Dedalus Labs, Inc.

"""Tiki tensors: an MLX array Engine composed with a CuTe Layout.

``from_array`` pairs a flattened array with its dense right-major layout.
``realize`` turns an affine layout back into one zero-copy MLX view through
``as_strided``. ``unsqueeze``, ``squeeze``, and ``expand`` follow zop's
trailing-axis broadcasting: expanded axes get stride 0, nothing is allocated,
and an incompatible extent is a ``LayoutError``, never a runtime guess.
"""

from math import prod
from typing import Any

import mlx.core as mx
from mlx.tiki._pycute import Layout, Tensor, flatten
from mlx.tiki.composed import ComposedLayout
from mlx.tiki.engine import ArrayEngine
from mlx.tiki.layout import LayoutError


def right_major_layout(shape: tuple[int, ...]) -> Layout:
    """Dense layout whose final axis has unit stride, matching MLX and NumPy."""
    strides = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
    return Layout(shape, strides)


def from_array(array: mx.array) -> Tensor:
    """View ``array`` as a tensor over its flattened storage.

    The base is ``array.reshape(-1)``. MLX cannot report whether a lazy view is
    contiguous, so a non-contiguous input is copied by that reshape.
    """
    return Tensor(
        ArrayEngine(array.reshape(-1)), right_major_layout(tuple(array.shape))
    )


def realize(tensor: Tensor) -> mx.array:
    """One zero-copy MLX view of an affine layout over an ``ArrayEngine``.

    Layout modes become the view's axes, leaf by leaf. Composed layouts have no
    strides and cannot be a view; they stay kernel-side.
    """
    if isinstance(tensor.layout, ComposedLayout):
        raise LayoutError(
            f"cannot realize a composed layout as a view: {tensor.layout}"
        )
    engine = tensor.accessor
    if not isinstance(engine, ArrayEngine):
        raise LayoutError(f"realize needs an ArrayEngine, got {type(engine).__name__}")
    shape = tuple(int(extent) for extent in flatten(tensor.layout.shape))
    strides = tuple(int(stride) for stride in flatten(tensor.layout.stride))
    return mx.as_strided(
        engine.base, shape=shape, strides=strides, offset=engine.offset
    )


def unsqueeze(tensor: Tensor, axis: int) -> Tensor:
    """Insert an extent-1, stride-0 mode before ``axis`` (0 through rank)."""
    shape, stride = _flat_modes(tensor.layout)
    if not 0 <= axis <= len(shape):
        raise LayoutError(f"unsqueeze axis {axis} outside 0..{len(shape)}")
    shape.insert(axis, 1)
    stride.insert(axis, 0)
    return Tensor(tensor.accessor, Layout(tuple(shape), tuple(stride)))


def squeeze(tensor: Tensor, axis: int) -> Tensor:
    """Remove mode ``axis``, which must have extent 1."""
    shape, stride = _flat_modes(tensor.layout)
    if not 0 <= axis < len(shape):
        raise LayoutError(f"squeeze axis {axis} outside 0..{len(shape) - 1}")
    if shape[axis] != 1:
        raise LayoutError(f"squeeze axis {axis} has extent {shape[axis]}, not 1")
    del shape[axis], stride[axis]
    return Tensor(tensor.accessor, Layout(tuple(shape), tuple(stride)))


def expand(tensor: Tensor, target: tuple[int, ...]) -> Tensor:
    """Broadcast to ``target`` by the trailing-axis rule, with stride 0 on expanded axes.

    A leading axis may be prepended; an extent of 1 may grow; any other change
    is a ``LayoutError``. The target is exact: there is no ``-1`` sentinel.
    """
    shape, stride = _flat_modes(tensor.layout)
    if len(target) < len(shape):
        raise LayoutError(f"expand target {target} has fewer axes than {tuple(shape)}")
    lead = len(target) - len(shape)
    shape = [1] * lead + shape
    stride = [0] * lead + stride
    for axis, (extent, wanted) in enumerate(zip(shape, target)):
        if extent == wanted:
            continue
        if extent != 1:
            raise LayoutError(
                f"expand axis {axis}: extent {extent} cannot become {wanted}"
            )
        shape[axis], stride[axis] = wanted, 0
    return Tensor(tensor.accessor, Layout(tuple(shape), tuple(stride)))


def _flat_modes(layout: Any) -> tuple[list[int], list[int]]:
    if isinstance(layout, ComposedLayout):
        raise LayoutError("broadcasting needs an affine layout")
    return [int(extent) for extent in flatten(layout.shape)], [
        int(stride) for stride in flatten(layout.stride)
    ]
