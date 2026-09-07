# Copyright © 2026 Dedalus Labs, Inc.

"""Composed layouts: ``outer o {offset} o inner``, zop's form for nonlinear maps.

A swizzle mixes index bits with XOR. The composed layout keeps the exact
outer map, internal offset, and inner coordinate domain:
``layout(coordinate) == outer(offset + inner(coordinate))``. Slicing must keep
the fixed contribution inside the composition, because moving it through a
nonlinear outer map changes addresses. ``slice_and_offset`` returns the residual
layout and an external Engine displacement that together preserve every parent
address, which ``test_tiki_layout`` checks coordinate by coordinate.

This mirrors zop's Rust bootstrap (``src/layout/expression.rs``) and CUTLASS's
``swizzle_layout.hpp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import SupportsIndex, TypeAlias, cast

from mlx.tiki._layout import LayoutError
from mlx.tiki._pycute import Layout, LayoutBase, Shape, rank, size
from mlx.tiki._pycute.typedefs import Coord, StrideScalar
from mlx.tiki.swizzle import Swizzle

Coordinate: TypeAlias = int | None | slice | tuple["Coordinate", ...]


def check_swizzle(swizzle: Swizzle) -> Swizzle:
    """Require the Rust-validated indexing transform at this boundary."""
    if not isinstance(swizzle, Swizzle):
        raise LayoutError("composition requires a validated Tiki Swizzle")
    return swizzle


@dataclass(frozen=True)
class ComposedLayout(LayoutBase):
    """An index transform composed with an internal offset and a layout domain."""

    outer: Swizzle | Layout | ComposedLayout
    offset: int
    inner: Layout | ComposedLayout

    def __post_init__(self) -> None:
        if not isinstance(self.outer, (Swizzle, Layout, ComposedLayout)):
            raise LayoutError("composition outer must be a Swizzle or a layout")
        if not isinstance(self.inner, (Layout, ComposedLayout)):
            raise LayoutError("composition inner must supply a layout domain")
        if type(self.offset) is not int:
            raise LayoutError("composition offset must be an integer")

    def __call__(self, *coordinate: Coordinate) -> int:
        """Evaluate ``outer(offset + inner(coordinate))`` without accessing storage."""
        argument = self.offset + _evaluate(self.inner, coordinate)
        if isinstance(self.outer, Swizzle):
            return self.outer(argument)
        return _evaluate(self.outer, (argument,))

    def swizzle(self, transform: Swizzle, offset: int = 0) -> "ComposedLayout":
        """Compose another index transform without changing storage ownership."""
        return ComposedLayout(check_swizzle(transform), offset, self)

    def _offset_and_slice(
        self, coordinate: Coordinate
    ) -> tuple[int, "Layout | ComposedLayout"]:
        residual, delta = slice_and_offset(coordinate, self)
        if rank(residual) == 0:
            return delta + _evaluate(residual, ()), Layout((), ())
        return delta, residual

    @property
    def shape(self) -> Shape:
        return self.inner.shape

    @property
    def stride(self) -> None:
        raise LayoutError("a composed layout has no stride. Require an affine layout")

    def __str__(self) -> str:
        return f"{self.outer} o {{{self.offset}}} o {self.inner}"


def slice_and_offset(
    coordinate: Coordinate, layout: "Layout | ComposedLayout"
) -> tuple["Layout | ComposedLayout", int]:
    """Fix the named coordinates; ``None`` keeps a mode free.

    Returns ``(residual, engine_delta)`` with
    ``parent(fixed, free) == engine_delta + residual(free)``. An affine layout
    moves the fixed contribution outside as the Engine displacement. A composed
    layout keeps it inside the composition and reports zero displacement.
    """
    if not isinstance(layout, ComposedLayout):
        offset, residual = layout._offset_and_slice(cast(Coord, coordinate))
        return residual, _offset(offset)
    residual, delta = slice_and_offset(coordinate, layout.inner)
    return ComposedLayout(layout.outer, layout.offset + delta, residual), 0


def _offset(value: StrideScalar | SupportsIndex) -> int:
    if not isinstance(value, SupportsIndex):
        raise LayoutError(
            "composition requires integer offset addition, not XOR or coordinate addition"
        )
    return index(value)


def _evaluate(
    layout: Layout | ComposedLayout, coordinate: tuple[Coordinate, ...]
) -> int:
    if isinstance(layout, ComposedLayout):
        return layout(*coordinate)
    # PyCuTe's coordinate ABCs register Python integers dynamically.
    return _offset(layout(*cast(tuple[Coord, ...], coordinate)))


def composed_size(layout: "Layout | ComposedLayout") -> int:
    return size(layout.inner if isinstance(layout, ComposedLayout) else layout)


def composed_rank(layout: "Layout | ComposedLayout") -> int:
    return rank(layout.inner if isinstance(layout, ComposedLayout) else layout)
