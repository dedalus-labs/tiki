# Copyright © 2026 Dedalus Labs, Inc.

"""Composed layouts: ``outer o {offset} o inner``, zop's form for nonlinear maps.

A swizzle is a bit permutation, so it has no stride tree. zop keeps the exact
composition of an outer map, an internal offset, and an inner layout:
``layout(coordinate) == outer(offset + inner(coordinate))``. Slicing must keep
the fixed contribution inside the composition, because moving it through a
nonlinear outer map changes addresses. ``slice_and_offset`` returns the residual
layout and an external Engine displacement that together preserve every parent
address, which ``test_tiki_layout`` checks coordinate by coordinate.

This mirrors zop's Rust bootstrap (``src/layout/expression.rs``) and CUTLASS's
``swizzle_layout.hpp``.
"""

from dataclasses import dataclass
from typing import Any

from mlx.tiki._pycute import Layout, LayoutBase, Swizzle, rank, size
from mlx.tiki.layout import LayoutError

Coordinate = Any


def check_swizzle(swizzle: Swizzle) -> Swizzle:
    """CUTLASS requires the two bit fields not to overlap: ``abs(shift) >= bits``.

    PyCuTe checks only negative shifts and zop's bootstrap checks only the
    highest bit, so both accept ``Swizzle(1, 0, 0)``, which maps 0 and 1 to
    the same index. Tiki rejects it.
    """
    if swizzle.bits and abs(swizzle.shift) < swizzle.bits:
        raise LayoutError(
            f"swizzle fields overlap: bits={swizzle.bits} shift={swizzle.shift}; CUTLASS requires abs(shift) >= bits"
        )
    return swizzle


@dataclass(frozen=True)
class ComposedLayout(LayoutBase):
    outer: Swizzle
    offset: int
    inner: "Layout | ComposedLayout"

    def __post_init__(self) -> None:
        check_swizzle(self.outer)

    def __call__(self, *coordinate: Coordinate) -> int:
        index = self.offset + self.inner(*coordinate)
        if index < 0:
            raise LayoutError(f"swizzle input {index} must be nonnegative")
        return self.outer(index)

    @property
    def shape(self) -> Any:
        return self.inner.shape

    @property
    def stride(self) -> Any:
        raise LayoutError("a composed layout has no stride; require an affine layout")

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
        offset, residual = layout._offset_and_slice(coordinate)
        return residual, int(offset)
    residual, delta = slice_and_offset(coordinate, layout.inner)
    return ComposedLayout(layout.outer, layout.offset + delta, residual), 0


def composed_size(layout: "Layout | ComposedLayout") -> int:
    return size(layout.inner if isinstance(layout, ComposedLayout) else layout)


def composed_rank(layout: "Layout | ComposedLayout") -> int:
    return rank(layout.inner if isinstance(layout, ComposedLayout) else layout)
