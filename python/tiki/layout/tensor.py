# Copyright © 2026 Dedalus Labs, Inc.

"""Tiki tensors: an Engine composed with a CuTe Layout.

``tensor[coordinate] == engine[layout(coordinate)]``. ``from_array`` pairs a
flattened Tiki array with its dense right-major layout.
``realize`` turns an affine layout back into one zero-copy Tiki view through
``as_strided``. ``unsqueeze``, ``squeeze``, and ``expand`` follow zop's
trailing-axis broadcasting: expanded axes get stride 0, nothing is allocated,
and an incompatible extent is a ``LayoutError``, never a runtime guess.
"""

import ctypes
from math import prod
from operator import index
from typing import Any, SupportsIndex, cast

import tiki as tk
from tiki.layout._cute import LayoutError
from tiki.layout.affine import Layout
from tiki.layout.composed import ComposedLayout, Coordinate, LayoutBase
from tiki.layout.engine import (
    Array,
    ArrayEngine,
    CType,
    Engine,
    ImplicitAccessor,
    MutableEngine,
)
from tiki.layout.tuples import Mode, Shape, Stride, flatten, make_basis_like, rank, size


class Tensor:
    """An Engine paired with a layout: ``tensor[c] == engine[layout(c)]``.

    A coordinate that leaves modes free with ``None`` or ``:`` returns a
    tensor over those modes that shares the Engine. Indexing never copies.
    """

    __slots__ = ("accessor", "layout")

    def __init__(self, accessor: Engine, layout: Layout | ComposedLayout) -> None:
        if not isinstance(accessor, Engine):
            raise LayoutError(f"a tensor needs an Engine, got {accessor!r}")
        if not isinstance(layout, LayoutBase):
            raise LayoutError(f"a tensor needs a layout, got {layout!r}")
        self.accessor = accessor
        self.layout = layout

    @property
    def shape(self) -> Shape:
        """Shape of the layout domain."""
        return self.layout.shape

    def __getitem__(self, coordinate: Coordinate) -> Any:
        """The element at ``coordinate``, or the tensor over the modes it leaves free."""
        offset, residual = self.layout._offset_and_slice(coordinate)
        if rank(residual) == 0:
            return self.accessor[offset]
        return Tensor(self.accessor + offset, residual)

    def __setitem__(self, coordinate: Coordinate, value: Any) -> None:
        """Write the element at ``coordinate``, which must name every mode."""
        if not isinstance(self.accessor, MutableEngine):
            raise LayoutError(f"{self.accessor!r} is a read-only Engine")
        offset, residual = self.layout._offset_and_slice(coordinate)
        if rank(residual) != 0:
            raise LayoutError(f"writing needs a whole coordinate, got {coordinate!r}")
        self.accessor[offset] = value

    def get(self, mode: Mode = ()) -> "Tensor":
        """The tensor over the mode at path ``mode``."""
        layout = self.layout
        for position in (mode,) if isinstance(mode, int) else mode:
            if not isinstance(layout, Layout):
                raise LayoutError("selecting a mode needs a stride layout")
            layout = layout[position]
        return Tensor(self.accessor, layout)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return bool(self.accessor == other.accessor and self.layout == other.layout)

    def __str__(self) -> str:
        return f"{self.accessor} o {self.layout}"

    def __repr__(self) -> str:
        return f"Tensor({self.accessor}, {self.layout})"


def make_tensor(layout: Layout | Shape, dtype: CType = ctypes.c_double) -> Tensor:
    """A tensor over a new ``Array`` of ``cosize(layout)`` elements."""
    if not isinstance(layout, Layout):
        if isinstance(layout, LayoutBase):
            raise LayoutError("make_tensor needs a stride layout")
        layout = Layout(layout)
    cosize = layout._native.coshape()
    if not isinstance(cosize, int):
        raise LayoutError(f"make_tensor needs an integer codomain, got {cosize}")
    return Tensor(Array(cosize, dtype), layout)


def identity_tensor(extents: Shape) -> Tensor:
    """The tensor whose element at each coordinate is that coordinate.

    Coordinate strides over an ``ImplicitAccessor`` allocate nothing.
    """
    return Tensor(ImplicitAccessor(0), Layout(extents, stride=make_basis_like(extents)))


def right_major_layout(shape: tuple[int, ...]) -> Layout:
    """Dense layout whose final axis has unit stride, matching Tiki and NumPy."""
    strides = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
    return Layout(shape, stride=strides)


def from_array(array: tk.array) -> Tensor:
    """View ``array`` as a tensor over its flattened storage.

    Noncontiguous inputs are normalized explicitly. Reshape alone can retain
    a strided vector, which cannot serve as an ``as_strided`` Engine.
    """
    return Tensor(
        ArrayEngine(tk.contiguous(array, allow_col_major=False).reshape(-1)),
        right_major_layout(tuple(array.shape)),
    )


def realize(tensor: Tensor) -> tk.array:
    """One zero-copy Tiki view of an affine layout over an ``ArrayEngine``.

    Layout modes become the view's axes, leaf by leaf. Composed layouts require
    a separate layout-aware consumer and cannot become an affine Tiki view.
    """
    if isinstance(tensor.layout, ComposedLayout):
        raise LayoutError(
            "cannot realize a composed layout as a view. Require an affine layout"
        )
    engine = tensor.accessor
    if not isinstance(engine, ArrayEngine):
        raise LayoutError(f"realize needs an ArrayEngine, got {type(engine).__name__}")
    try:
        shape = tuple(
            index(cast(SupportsIndex, e)) for e in flatten(tensor.layout.shape)
        )
        strides = tuple(
            index(cast(SupportsIndex, d)) for d in flatten(tensor.layout.stride)
        )
    except TypeError as error:
        raise LayoutError(
            "realize requires integer strides, not coordinate or XOR strides"
        ) from error
    if any(extent >= 2**31 for extent in shape):
        raise LayoutError(f"realize needs Tiki extents below 2**31, got {shape}")
    deltas = [(extent - 1) * stride for extent, stride in zip(shape, strides)]
    lower = engine.offset + sum(min(0, delta) for delta in deltas)
    upper = engine.offset + sum(max(0, delta) for delta in deltas)
    if lower < 0 or upper >= engine.base.size:
        raise LayoutError(
            f"layout addresses [{lower}, {upper}] outside Engine [0, {engine.base.size})"
        )
    return tk.as_strided(
        engine.base, shape=shape, strides=strides, offset=engine.offset
    )


def unsqueeze(tensor: Tensor, axis: int) -> Tensor:
    """Insert an extent-1, stride-0 mode before ``axis`` (0 through rank)."""
    shape, stride = _modes(tensor.layout)
    if not 0 <= axis <= len(shape):
        raise LayoutError(f"unsqueeze axis {axis} outside 0..{len(shape)}")
    shape.insert(axis, 1)
    stride.insert(axis, 0)
    return Tensor(tensor.accessor, Layout(tuple(shape), stride=tuple(stride)))


def squeeze(tensor: Tensor, axis: int) -> Tensor:
    """Remove mode ``axis``, which must have extent 1."""
    shape, stride = _modes(tensor.layout)
    if not 0 <= axis < len(shape):
        raise LayoutError(f"squeeze axis {axis} outside 0..{len(shape) - 1}")
    if size(shape[axis]) != 1:
        raise LayoutError(f"squeeze axis {axis} has extent {shape[axis]}, not 1")
    del shape[axis], stride[axis]
    return Tensor(tensor.accessor, Layout(tuple(shape), stride=tuple(stride)))


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
    ones: list[Shape] = [1] * lead
    zeros: list[Stride] = [0] * lead
    shape, stride = ones + shape, zeros + stride
    for axis, (mode, wanted) in enumerate(zip(shape, target)):
        extent = size(mode)
        if extent == wanted:
            continue
        if extent != 1:
            raise LayoutError(
                f"expand axis {axis}: extent {extent} cannot become {wanted}"
            )
        shape[axis], stride[axis] = wanted, 0
    return Tensor(tensor.accessor, Layout(tuple(shape), stride=tuple(stride)))


def _modes(layout: Layout | ComposedLayout) -> tuple[list[Shape], list[Stride]]:
    """Preserve top-level axes instead of promoting nested leaves to axes."""
    if isinstance(layout, ComposedLayout):
        raise LayoutError("broadcasting needs an affine layout")
    if isinstance(layout.shape, tuple):
        return list(layout.shape), list(cast(tuple[Stride, ...], layout.stride))
    return [layout.shape], [layout.stride]
