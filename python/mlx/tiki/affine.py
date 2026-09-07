# Copyright © 2026 Dedalus Labs, Inc.

"""PyCuTe stride layouts with the same composition entrypoint as nonlinear maps."""

from types import NotImplementedType
from typing import cast

from mlx.tiki._pycute import Layout as ReferenceLayout
from mlx.tiki._pycute import LayoutBase
from mlx.tiki._pycute.typedefs import Coord, Integer, StrideScalar
from mlx.tiki.composed import ComposedLayout, Coordinate, check_swizzle
from mlx.tiki.swizzle import Swizzle


class Layout(ReferenceLayout):
    """Map a hierarchical coordinate domain through a congruent stride tree.

    The default strides are column-major. Explicit integer strides can be
    signed or zero. ``swizzle`` composes an index transform with this map and
    retains its domain. It does not create a different tensor or allocate data.
    """

    def swizzle(self, transform: Swizzle, offset: int = 0) -> ComposedLayout:
        """Construct ``transform(offset + self(coordinate))``."""
        return ComposedLayout(check_swizzle(transform), offset, self)

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
