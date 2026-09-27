# Copyright © 2026 Dedalus Labs, Inc.

"""Layouts: hierarchical ``Shape:Stride`` coordinate maps and their algebra.

tiki-cute, Tiki's Rust implementation of the CuTe layout algebra, computes
every operation here. ``compose`` also retains nonlinear transforms and their
internal offsets. ``basis`` is ``E``, and ``cosize`` is the size of
``coshape``, not a storage-bounds proof.
"""

from collections.abc import Callable, Iterable
from itertools import zip_longest
from typing import TypeAlias, cast

from tiki.layout import _cute
from tiki.layout._cute import F2, LayoutError
from tiki.layout.affine import Layout
from tiki.layout.composed import ComposedLayout, LayoutBase
from tiki.layout.engine import (
    Array,
    Engine,
    ImplicitAccessor,
    MutableEngine,
    Ptr,
    TransformAccessor,
)
from tiki.layout.swizzle import Swizzle
from tiki.layout.tensor import Tensor, identity_tensor, make_tensor
from tiki.layout.tuples import (
    E,
    Mode,
    Shape,
    compatible,
    congruent,
    crd2idx,
    depth,
    flatten,
    idx2crd,
    modes,
    rank,
    size,
    unflatten,
)

__all__ = [
    "Array",
    "E",
    "Engine",
    "F2",
    "ImplicitAccessor",
    "Layout",
    "LayoutError",
    "MutableEngine",
    "Ptr",
    "Swizzle",
    "Tensor",
    "TransformAccessor",
    "basis",
    "blocked_product",
    "coalesce",
    "compatible",
    "complement",
    "compose",
    "congruent",
    "coshape",
    "cosize",
    "crd2idx",
    "depth",
    "flatten",
    "identity_tensor",
    "idx2crd",
    "is_layout",
    "left_inverse",
    "logical_divide",
    "logical_product",
    "make_layout",
    "make_tensor",
    "nullspace",
    "raked_product",
    "rank",
    "recast",
    "right_inverse",
    "size",
    "unflatten",
    "zipped_divide",
]

basis = E

Tiler: TypeAlias = int | Layout | None | tuple["Tiler", ...] | list["Tiler"]
NativeTiler: TypeAlias = int | _cute.Layout | None | tuple["NativeTiler", ...]
Operand: TypeAlias = Layout | Tensor | Shape


def is_layout(value: object) -> bool:
    """Whether ``value`` is a stride layout or a composed layout."""
    return isinstance(value, LayoutBase)


def _stride_layout(value: object, operation: str) -> Layout:
    """The stride layout an operation acts on. An extent or tuple tiles as ``Tiler`` does."""
    if isinstance(value, Layout):
        return value
    if isinstance(value, ComposedLayout) or (
        isinstance(value, Tensor) and isinstance(value.layout, ComposedLayout)
    ):
        raise LayoutError(
            f"{operation} requires a stride layout. Transform the domain before composing"
        )
    if isinstance(value, (int, tuple, list)) and not isinstance(value, bool):
        return Layout._of(_cute.Layout.from_tiler(_tiler(value)))
    raise LayoutError(f"{operation} requires a layout, got {type(value).__name__}")


def _tiler(tiler: object) -> NativeTiler:
    """A tiler in the binding's terms: each layout becomes its tiki-cute handle."""
    if isinstance(tiler, Layout):
        return tiler._native
    if isinstance(tiler, (tuple, list)):
        return tuple(_tiler(item) for item in tiler)
    if isinstance(tiler, LayoutBase):
        raise LayoutError("a tiler requires stride layouts. Transform the domain first")
    if tiler is None or (isinstance(tiler, int) and not isinstance(tiler, bool)):
        return tiler
    raise LayoutError(
        f"a tiler is a layout, an extent, or a tuple of them, got {tiler!r}"
    )


def _at_mode(
    layout: Layout, mode: Mode, operation: Callable[[Layout], Layout]
) -> Layout:
    """Apply ``operation`` to the mode at path ``mode`` and keep every other mode."""
    path = modes(mode)
    if not path:
        return operation(layout)
    head, rest = path[0], path[1:]
    if not 0 <= head < rank(layout):
        raise LayoutError(f"{layout} has no mode {head}")
    if not isinstance(layout.shape, tuple):
        return _at_mode(layout, rest, operation)
    parts = [layout[i] for i in range(rank(layout))]
    parts[head] = _at_mode(parts[head], rest, operation)
    return make_layout(parts)


def _apply(
    value: Operand,
    operation: str,
    native: Callable[[_cute.Layout], _cute.Layout],
    mode: Mode = (),
) -> Layout | Tensor:
    """Apply a tiki-cute operation to a layout, or to a tensor's layout over the same Engine."""
    if isinstance(value, Tensor) and isinstance(value.layout, Layout):
        return Tensor(
            value.accessor, cast(Layout, _apply(value.layout, operation, native, mode))
        )
    layout = _stride_layout(value, operation)
    return _at_mode(layout, mode, lambda part: Layout._of(native(part._native)))


def make_layout(layouts: Iterable[Layout]) -> Layout:
    """Concatenate layouts: each becomes one top-level mode of the result."""
    parts = list(layouts)
    if not all(isinstance(part, Layout) for part in parts):
        raise LayoutError("make_layout requires stride layouts")
    return Layout._of(_cute.Layout.from_modes([part._native for part in parts]))


def _coalesce(layout: Layout, profile: object) -> Layout:
    if profile is None:
        return layout
    if isinstance(profile, (tuple, list)):
        if rank(layout) < len(profile):
            raise LayoutError(
                f"coalesce profile {profile} has more modes than {layout}"
            )
        parts = zip_longest(range(rank(layout)), profile)
        return make_layout([_coalesce(layout[i], part) for i, part in parts])
    return Layout._of(layout._native.coalesce())


def coalesce(
    value: Operand, profile: object = 1, *, mode: Mode = ()
) -> Layout | Tensor:
    """The flat layout with the same function on every index of the domain.

    A tuple ``profile`` coalesces each mode it names, and ``None`` leaves a
    mode as it is.
    """
    if isinstance(value, Tensor) and isinstance(value.layout, Layout):
        return Tensor(
            value.accessor, cast(Layout, coalesce(value.layout, profile, mode=mode))
        )
    layout = _stride_layout(value, "coalesce")
    return _at_mode(layout, mode, lambda part: _coalesce(part, profile))


def compose(
    outer: LayoutBase | Tensor | Swizzle | Tiler,
    inner: LayoutBase | Tiler,
    *,
    offset: int = 0,
    mode: Mode = (),
) -> LayoutBase | Tensor:
    """Compose coordinate maps, retaining offsets inside nonlinear transforms.

    Integer-affine composition runs in tiki-cute. A nonlinear operand or an
    explicit offset stays an inspectable expression over the inner domain.
    """
    if type(offset) is not int:
        raise LayoutError("composition offset must be an integer")
    if (
        isinstance(outer, (Swizzle, ComposedLayout))
        or isinstance(inner, ComposedLayout)
        or offset != 0
    ):
        if mode != ():
            raise LayoutError(
                "compose the selected domain before applying a nonlinear transform"
            )
        if not isinstance(outer, (Swizzle, LayoutBase)):
            raise LayoutError("composition outer must be a Swizzle or a layout")
        if not isinstance(inner, LayoutBase):
            raise LayoutError("composition inner must supply a layout domain")
        # LayoutBase has exactly two subclasses, Layout and ComposedLayout.
        return ComposedLayout(
            outer=cast(Swizzle | Layout | ComposedLayout, outer),
            offset=offset,
            inner=cast(Layout | ComposedLayout, inner),
        )
    if inner is None:
        return cast(Layout | Tensor, outer)
    tiler = _tiler(inner)
    return _apply(cast(Operand, outer), "compose", lambda a: a.compose(tiler), mode)


def logical_divide(value: Operand, tiler: Tiler, *, mode: Mode = ()) -> Layout | Tensor:
    """Split ``value`` by the tile ``tiler``: mode 0 is the tile, mode 1 indexes the tiles."""
    if tiler is None:
        return cast(Layout | Tensor, value)
    native = _tiler(tiler)
    return _apply(value, "logical_divide", lambda a: a.logical_divide(native), mode)


def zipped_divide(value: Operand, tiler: Tiler, *, mode: Mode = ()) -> Layout | Tensor:
    """Divide as ``logical_divide`` does, with every tile mode gathered into mode 0."""
    if tiler is None:
        return cast(Layout | Tensor, value)
    native = _tiler(tiler)
    return _apply(value, "zipped_divide", lambda a: a.zipped_divide(native), mode)


def logical_product(value: Operand, tiler: Tiler, *, mode: Mode = ()) -> Layout:
    """One copy of ``value`` per coordinate of ``tiler``, placed where ``value`` leaves room."""
    layout = _stride_layout(value, "logical_product")
    if tiler is None:
        return layout
    native = _tiler(tiler)
    return cast(
        Layout,
        _apply(layout, "logical_product", lambda a: a.logical_product(native), mode),
    )


def _tile(tile: object, operation: str) -> _cute.Layout:
    return _stride_layout(tile, operation)._native


def blocked_product(value: Operand, tile: Layout | int) -> Layout:
    """The product that keeps each copy of ``value`` contiguous in every mode."""
    native = _tile(tile, "blocked_product")
    return cast(
        Layout, _apply(value, "blocked_product", lambda a: a.blocked_product(native))
    )


def raked_product(value: Operand, tile: Layout | int) -> Layout:
    """The product that interleaves the copies of ``value`` in every mode."""
    native = _tile(tile, "raked_product")
    return cast(
        Layout, _apply(value, "raked_product", lambda a: a.raked_product(native))
    )


def complement(value: Operand, extend: Shape | None = None) -> Layout:
    """The ordered layout that fills the offsets ``value`` leaves free, grown to ``extend``."""
    cotarget = extend if extend else None
    return cast(Layout, _apply(value, "complement", lambda a: a.complement(cotarget)))


def right_inverse(value: Operand) -> Layout:
    """The largest layout ``R`` with ``value(R(i)) == i`` on its domain."""
    return cast(Layout, _apply(value, "right_inverse", lambda a: a.right_inverse()))


def left_inverse(value: Operand) -> Layout:
    """A layout ``L`` with ``value(L(value(i))) == value(i)``."""
    return cast(Layout, _apply(value, "left_inverse", lambda a: a.left_inverse()))


def nullspace(value: Operand) -> Layout:
    """The layout of the coordinates that ``value`` maps to 0."""
    return cast(Layout, _apply(value, "nullspace", lambda a: a.nullspace()))


def recast(layout: Operand, scale: Shape) -> Layout:
    """The layout over elements ``scale`` times as wide, one integer factor per codomain axis."""
    return cast(Layout, _apply(layout, "recast", lambda a: a.recast(scale)))


def coshape(layout: Layout) -> Shape:
    """Shape of the codomain: one past the largest offset in each codomain mode."""
    if not isinstance(layout, Layout):
        raise LayoutError("coshape requires a stride layout")
    return layout._native.coshape()


def cosize(layout: Layout) -> int:
    """Size of the codomain. This bounds offsets. It is not a storage-bounds proof."""
    return size(coshape(layout))
