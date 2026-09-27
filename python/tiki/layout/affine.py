# Copyright © 2026 Dedalus Labs, Inc.

"""Stride layouts, held as tiki-cute values, with the same composition entrypoint as nonlinear maps."""

from types import NotImplementedType
from typing import cast

from tiki.layout import _cute
from tiki.layout._cute import F2, ArithTuple
from tiki.layout.composed import (
    ComposedLayout,
    Coordinate,
    LayoutBase,
    check_swizzle,
    cute,
    describe,
)
from tiki.layout.swizzle import Swizzle, notation
from tiki.layout.tuples import Coord, Offset, Shape, Stride


class Layout(LayoutBase):
    """Map a hierarchical coordinate domain through a congruent stride tree.

    Construct it as ``Layout(shape, stride=...)``. The default strides are
    column-major. Explicit integer strides can be signed or zero. ``swizzle``
    composes an index transform with this map and retains its domain. It does
    not create a different tensor or allocate data.

    tiki-cute checks the layout when it is built: every extent is positive
    and the stride is congruent to the shape. A layout is immutable.
    """

    __slots__ = ("_native",)
    _native: _cute.Layout

    def __init__(self, shape: Shape, *, stride: Stride = 1) -> None:
        self._native = _cute.Layout(shape, stride)

    @classmethod
    def _of(cls, native: _cute.Layout) -> "Layout":
        """Wrap a layout tiki-cute already checked."""
        layout = cls.__new__(cls)
        layout._native = native
        return layout

    @property
    def shape(self) -> Shape:
        return self._native.shape

    @property
    def stride(self) -> Stride:
        return self._native.stride

    def __call__(self, *coordinate: Coord | F2 | ArithTuple) -> Offset:
        """The offset of an index, a coordinate per mode, or a natural coordinate."""
        if len(coordinate) == 1:
            return self._native(coordinate[0])
        return self._native(cast(Coord, coordinate))

    def __getitem__(self, mode: int) -> "Layout":
        """Mode ``mode`` of the layout. A negative mode counts from the last."""
        rank = self._native.rank
        position = mode + rank if mode < 0 else mode
        if not 0 <= position < rank:
            raise IndexError(f"mode {mode} is out of range for {self}")
        return Layout._of(self._native.get((position,)))

    def _offset_and_slice(self, coordinate: Coordinate) -> tuple[Offset, "Layout"]:
        offset, residual = self._native.slice(coordinate)
        return offset, Layout._of(residual)

    def describe(self) -> str:
        """One coordinate per line with its extent and stride, then the index formula."""
        return describe(self)

    def __repr__(self) -> str:
        return f"Layout(shape={self.shape}, stride={self.stride})"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        """``format(layout, "cute")`` is CuTe's ``shape:stride``."""
        return notation(self, spec, cute(self))

    def swizzle(self, transform: Swizzle, offset: int = 0) -> ComposedLayout:
        """Construct ``transform(offset + self(coordinate))``."""
        return ComposedLayout(outer=check_swizzle(transform), offset=offset, inner=self)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if isinstance(other, Layout):
            return self._native == other._native
        if isinstance(other, LayoutBase):
            return False
        return NotImplemented
