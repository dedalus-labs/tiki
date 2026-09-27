# Copyright © 2026 Dedalus Labs, Inc.

"""Hierarchical tuples: the shapes, strides, and coordinates of layouts.

A tuple or list is a node and anything else is a leaf, as in CuTe. These
functions read the structure of such values. The functions that compute with
extents, ``idx2crd``, ``crd2idx``, and ``compatible``, run in tiki-cute.
"""

from collections.abc import Iterator
from math import prod
from typing import Protocol, TypeAlias, TypeVar, cast

from tiki.layout import _cute
from tiki.layout._cute import F2, ArithTuple

Shape: TypeAlias = int | tuple["Shape", ...]
Offset: TypeAlias = int | F2 | ArithTuple
Stride: TypeAlias = Offset | tuple["Stride", ...]
Coord: TypeAlias = int | tuple["Coord", ...]
XorCoord: TypeAlias = F2 | tuple["XorCoord", ...]
Mode: TypeAlias = int | tuple[int, ...]

Leaf = TypeVar("Leaf")
Tree: TypeAlias = object


class HasShape(Protocol):
    @property
    def shape(self) -> Shape: ...


def is_tuple(value: object) -> bool:
    """A tuple or list is a node of a hierarchical tuple."""
    return isinstance(value, (tuple, list))


def modes(mode: Mode) -> tuple[int, ...]:
    """A mode path: one index per level. An integer is a path of one index."""
    return mode if isinstance(mode, tuple) else (mode,)


def get(value: Tree, mode: Mode = ()) -> Tree:
    """The subtree at ``mode``, one index per level."""
    for index in modes(mode):
        value = cast(tuple[Tree, ...], value)[index]
    return value


def shape(value: HasShape | Shape, mode: Mode = ()) -> Shape:
    """The shape of a layout or tensor, or the value itself for a bare shape."""
    whole = value.shape if hasattr(value, "shape") else value
    return cast(Shape, get(whole, mode))


def leaves(value: Tree) -> Iterator[Tree]:
    if is_tuple(value):
        for item in cast(tuple[Tree, ...], value):
            yield from leaves(item)
    else:
        yield value


def flatten(value: Tree) -> tuple[Tree, ...]:
    """The leaves in order, as one flat tuple."""
    return tuple(leaves(value))


def unflatten(values: Iterator[Leaf], profile: Tree) -> Tree:
    """Rebuild the structure of ``profile`` from leaves taken in order."""
    if is_tuple(profile):
        return tuple(
            unflatten(values, item) for item in cast(tuple[Tree, ...], profile)
        )
    return next(values)


def size(value: HasShape | Shape, *, mode: Mode = ()) -> int:
    """Number of coordinates: the product of the extents."""
    return prod(cast(Iterator[int], leaves(shape(value, mode))))


def rank(value: HasShape | Shape, *, mode: Mode = ()) -> int:
    """Number of top-level modes. A leaf has rank 1."""
    whole = shape(value, mode)
    return len(whole) if isinstance(whole, tuple) else 1


def depth(value: HasShape | Shape, *, mode: Mode = ()) -> int:
    """Levels of nesting. A leaf has depth 0."""
    whole = shape(value, mode)
    if not isinstance(whole, tuple):
        return 0
    return 1 + max((depth(item) for item in whole), default=0)


def profile(value: Tree) -> Tree:
    """The tree whose structure congruence compares: a layout's shape, or the value."""
    return value.shape if hasattr(value, "shape") else value


def congruent(a: Tree, b: Tree) -> bool:
    """Whether ``a`` and ``b`` have the same structure, leaf for leaf."""
    a, b = profile(a), profile(b)
    if is_tuple(a) and is_tuple(b):
        a_modes, b_modes = cast(tuple[Tree, ...], a), cast(tuple[Tree, ...], b)
        return len(a_modes) == len(b_modes) and all(
            congruent(x, y) for x, y in zip(a_modes, b_modes)
        )
    return not (is_tuple(a) or is_tuple(b))


def compatible(a: HasShape | Shape, b: HasShape | Shape) -> bool:
    """Whether every coordinate of ``a`` is a coordinate of ``b``."""
    return _cute.compatible(shape(a), shape(b))


def idx2crd(index: Coord | F2 | None, extents: Shape) -> Coord | XorCoord:
    """The natural coordinate of ``index`` in ``extents``, first mode fastest.

    ``None`` is the zero coordinate. An ``F2`` index splits carry-lessly.
    """
    if index is None:
        return cast(Coord, unflatten(iter(lambda: 0, None), extents))
    if isinstance(index, F2):
        return _cute.idx2crd(index, extents)
    return _cute.idx2crd(index, extents)


def crd2idx(coordinate: Coord, extents: Shape) -> int:
    """The index of ``coordinate`` in ``extents``, the inverse of ``idx2crd``."""
    return _cute.crd2idx(coordinate, extents)


def E(*mode: int) -> int | ArithTuple:
    """The unit basis element along codomain axis ``mode``: ``E()`` is 1, ``E(1)`` is ``(0, 1)``."""
    return _cute.basis(mode)


def make_basis_like(extents: Tree, mode: tuple[int, ...] = ()) -> Stride:
    """One basis element per leaf of ``extents``, ``E(path)`` at each leaf path."""
    if is_tuple(extents):
        items = cast(tuple[Tree, ...], extents)
        return tuple(make_basis_like(item, (*mode, i)) for i, item in enumerate(items))
    return E(*mode)
