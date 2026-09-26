# Copyright © 2026 Dedalus Labs, Inc.

"""Tiki tensors: an MLX array Engine composed with a CuTe Layout.

``from_array`` pairs a flattened array with its dense right-major layout.
``realize`` turns an affine layout back into one zero-copy MLX view through
``as_strided``. ``unsqueeze``, ``squeeze``, and ``expand`` follow zop's
trailing-axis broadcasting: expanded axes get stride 0, nothing is allocated,
and an incompatible extent is a ``LayoutError``, never a runtime guess.
"""

from math import prod
from operator import index

import mlx.core as mx
from mlx.tiki._pycute import Shape, Stride, Tensor, flatten, size
from mlx.tiki.affine import Layout
from mlx.tiki.composed import ComposedLayout
from mlx.tiki.engine import ArrayEngine
from mlx.tiki.layout import LayoutError


def right_major_layout(shape: tuple[int, ...]) -> Layout:
    """Dense layout whose final axis has unit stride, matching MLX and NumPy."""
    strides = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
    return Layout(shape, strides)


def from_array(array: mx.array) -> Tensor:
    """View ``array`` as a tensor over its flattened storage.

    Noncontiguous inputs are normalized explicitly. Reshape alone can retain
    a strided vector, which cannot serve as an ``as_strided`` Engine.
    """
    return Tensor(
        ArrayEngine(mx.contiguous(array, allow_col_major=False).reshape(-1)),
        right_major_layout(tuple(array.shape)),
    )


def realize(tensor: Tensor) -> mx.array:
    """One zero-copy MLX view of an affine layout over an ``ArrayEngine``.

    Layout modes become the view's axes, leaf by leaf. Composed layouts require
    a separate layout-aware consumer and cannot become an affine MLX view.
    """
    if isinstance(tensor.layout, ComposedLayout):
        raise LayoutError(
            f"cannot realize a composed layout as a view: {tensor.layout}"
        )
    engine = tensor.accessor
    if not isinstance(engine, ArrayEngine):
        raise LayoutError(f"realize needs an ArrayEngine, got {type(engine).__name__}")
    try:
        shape = tuple(index(extent) for extent in flatten(tensor.layout.shape))
        strides = tuple(index(stride) for stride in flatten(tensor.layout.stride))
    except TypeError as error:
        raise LayoutError(
            "realize requires integer strides, not coordinate or XOR strides"
        ) from error
    if len(shape) != len(strides) or any(
        extent < 0 or extent >= 2**31 for extent in shape
    ):
        raise LayoutError(
            f"realize needs matching nonnegative MLX extents and strides, got {shape}"
        )
    if any(not -(2**63) <= stride < 2**63 for stride in strides):
        raise LayoutError("realize strides must fit signed 64-bit indexing")
    if prod(shape) != 0:
        deltas = [(extent - 1) * stride for extent, stride in zip(shape, strides)]
        lower = engine.offset + sum(min(0, delta) for delta in deltas)
        upper = engine.offset + sum(max(0, delta) for delta in deltas)
        if lower < 0 or upper >= engine.base.size:
            raise LayoutError(
                f"layout addresses [{lower}, {upper}] outside Engine [0, {engine.base.size})"
            )
    return mx.as_strided(
        engine.base, shape=shape, strides=strides, offset=engine.offset
    )


def unsqueeze(tensor: Tensor, axis: int) -> Tensor:
    """Insert an extent-1, stride-0 mode before ``axis`` (0 through rank)."""
    shape, stride = _modes(tensor.layout)
    if not 0 <= axis <= len(shape):
        raise LayoutError(f"unsqueeze axis {axis} outside 0..{len(shape)}")
    shape.insert(axis, 1)
    stride.insert(axis, 0)
    return Tensor(tensor.accessor, Layout(tuple(shape), tuple(stride)))


def squeeze(tensor: Tensor, axis: int) -> Tensor:
    """Remove mode ``axis``, which must have extent 1."""
    shape, stride = _modes(tensor.layout)
    if not 0 <= axis < len(shape):
        raise LayoutError(f"squeeze axis {axis} outside 0..{len(shape) - 1}")
    if size(shape[axis]) != 1:
        raise LayoutError(f"squeeze axis {axis} has extent {shape[axis]}, not 1")
    del shape[axis], stride[axis]
    return Tensor(tensor.accessor, Layout(tuple(shape), tuple(stride)))


def expand(tensor: Tensor, target: tuple[int, ...]) -> Tensor:
    """Broadcast to ``target`` by the trailing-axis rule, with stride 0 on expanded axes.

    A leading axis may be prepended; an extent of 1 may grow; any other change
    is a ``LayoutError``. The target is exact: there is no ``-1`` sentinel.
    """
    shape, stride = _modes(tensor.layout)
    try:
        target = tuple(index(extent) for extent in target)
    except TypeError as error:
        raise LayoutError("expand target extents must be integers") from error
    if any(extent < 0 for extent in target):
        raise LayoutError(f"expand target extents must be nonnegative, got {target}")
    if len(target) < len(shape):
        raise LayoutError(f"expand target {target} has fewer axes than {tuple(shape)}")
    lead = len(target) - len(shape)
    shape = [1] * lead + shape
    stride = [0] * lead + stride
    for axis, (mode, wanted) in enumerate(zip(shape, target)):
        extent = size(mode)
        if extent == wanted:
            continue
        if extent != 1:
            raise LayoutError(
                f"expand axis {axis}: extent {extent} cannot become {wanted}"
            )
        shape[axis], stride[axis] = wanted, 0
    return Tensor(tensor.accessor, Layout(tuple(shape), tuple(stride)))


def _modes(layout: Layout | ComposedLayout) -> tuple[list[Shape], list[Stride]]:
    """Preserve top-level axes instead of promoting nested leaves to axes."""
    if isinstance(layout, ComposedLayout):
        raise LayoutError("broadcasting needs an affine layout")
    if isinstance(layout.shape, (tuple, list)):
        return list(layout.shape), list(layout.stride)
    return [layout.shape], [layout.stride]
