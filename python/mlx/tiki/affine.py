# Copyright © 2026 Dedalus Labs, Inc.

"""PyCuTe stride layouts with the same composition entrypoint as nonlinear maps."""

from types import NotImplementedType
from typing import cast

from mlx.tiki._pycute import Layout as ReferenceLayout
from mlx.tiki._pycute import LayoutBase
from mlx.tiki._pycute.typedefs import Coord, Integer, Shape, Stride, StrideScalar
from mlx.tiki.composed import ComposedLayout, Coordinate, check_swizzle, cute, describe
from mlx.tiki.swizzle import Swizzle, notation


class Layout(ReferenceLayout):
    """Map a hierarchical coordinate domain through a congruent stride tree.

    Construct it as ``Layout(shape, stride=...)``. The default strides are
    column-major. Explicit integer strides can be signed or zero. ``swizzle``
    composes an index transform with this map and retains its domain. It does
    not create a different tensor or allocate data.
    """

    def __init__(self, shape: Shape, *, stride: Stride = 1) -> None:
        super().__init__(shape, stride)

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

    def __getitem__(self, i: Integer | int) -> "Layout":
        # PyCuTe registers Python integers at runtime, beyond its static annotations.
        result = super().__getitem__(cast(Integer, i))
        return Layout._set(result.shape, result.stride)

    def _offset_and_slice(
        self, crd: Coord | Coordinate
    ) -> tuple[StrideScalar, "Layout"]:
        offset, result = super()._offset_and_slice(cast(Coord, crd))
        return offset, Layout._set(result.shape, result.stride)

    def __eq__(self, other: LayoutBase) -> bool | NotImplementedType:
        if isinstance(other, ComposedLayout):
            return False
        return super().__eq__(other)
