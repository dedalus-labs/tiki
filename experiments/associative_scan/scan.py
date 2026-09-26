"""Associative scan over MLX arrays, a port of ``jax.lax.associative_scan``.

The algorithm is Blelloch's recursive odd/even scan [BLE1990]: combine adjacent
pairs, scan the half-length sequence recursively, then fill the even positions
from the odd results. Work is O(n) combines and depth is O(log n). Every step is
an ordinary MLX array operation, so reverse-mode and forward-mode derivatives
come from MLX's transforms; no backward is registered here. This is also how
JAX differentiates its native GPU ``cumsum``: it re-expresses the kernel as
this tree and differentiates the tree.

Pairs are formed with stride-2 slices exactly as in JAX. That relies on the
Tiki fix to ``normalize_slice`` (a singleton strided slice records a
consistent stop and stride), so ``Slice::vjp`` and ``Slice::jvp`` are correct.
Derivative and vectorization tests skip on an MLX build without that fix.

[BLE1990] Blelloch, Guy E. 1990. "Prefix Sums and Their Applications."
CMU-CS-90-190.
"""

from collections.abc import Callable, Hashable
from typing import TypeAlias, TypeVar, cast

import mlx.core as mx
from mlx.utils import tree_map

Leaves = list[mx.array]
ArrayTree: TypeAlias = (
    mx.array | list["ArrayTree"] | tuple["ArrayTree", ...] | dict[Hashable, "ArrayTree"]
)
Tree = TypeVar("Tree", bound=ArrayTree)


def associative_scan(
    fn: Callable[[Tree, Tree], Tree],
    elems: Tree,
    reverse: bool = False,
    axis: int = 0,
) -> Tree:
    """Scan ``elems`` along ``axis`` with the associative binary operation ``fn``.

    ``fn(a, b)`` receives and returns pytrees shaped like ``elems`` and must be
    associative. Element ``k`` of the result is ``fn`` folded over the first
    ``k + 1`` elements; with ``reverse`` it is folded over the last ones. The
    combine order is preserved, so non-commutative operations are correct.
    """
    if not callable(fn):
        raise TypeError("associative_scan: fn must be callable")
    leaves = _flatten_like(elems, elems)
    _check_lengths(leaves, axis)
    if reverse:
        leaves = [_slice(leaf, slice(None, None, -1), axis) for leaf in leaves]

    def rebuild(values: Leaves) -> Tree:
        iterator = iter(values)
        return cast(Tree, tree_map(lambda _: next(iterator), elems))

    def combine(a: Leaves, b: Leaves) -> Leaves:
        result = _flatten_like(elems, fn(rebuild(a), rebuild(b)))
        for original, combined in zip(a, result):
            if combined.shape != original.shape or combined.dtype != original.dtype:
                raise ValueError(
                    "associative_scan: combine must preserve each leaf's shape and dtype"
                )
        return result

    scans = _scan(combine, leaves, axis)
    if reverse:
        scans = [_slice(scan, slice(None, None, -1), axis) for scan in scans]
    return rebuild(scans)


def _flatten_like(template: ArrayTree, value: ArrayTree) -> Leaves:
    """Match containers and dictionary keys without encoding them as strings."""
    if isinstance(template, mx.array):
        if not isinstance(value, mx.array):
            raise ValueError("associative_scan: combine replaced an array leaf")
        return [value]
    if isinstance(template, dict):
        if not isinstance(value, dict) or template.keys() != value.keys():
            raise ValueError("associative_scan: combine changed dictionary keys")
        return [
            leaf
            for key in template
            for leaf in _flatten_like(template[key], value[key])
        ]
    if isinstance(template, (tuple, list)):
        if type(template) is not type(value) or len(template) != len(value):
            raise ValueError("associative_scan: combine changed the tree structure")
        return [
            leaf
            for left, right in zip(template, value)
            for leaf in _flatten_like(left, right)
        ]
    raise TypeError("associative_scan: leaves must be MLX arrays")


def _scan(
    combine: Callable[[Leaves, Leaves], Leaves], elems: Leaves, axis: int
) -> Leaves:
    n = elems[0].shape[axis]
    if n < 2:
        return elems
    left, right = _pairs(elems, axis)
    odd = _scan(combine, combine(left, right), axis)
    if n % 2 == 0:
        even_sources = [_slice(leaf, slice(1, None), axis) for leaf in left]
        odd_prefix = [_slice(scan, slice(None, -1), axis) for scan in odd]
    else:
        even_sources = [
            mx.concatenate(
                [
                    _slice(leaf, slice(1, None), axis),
                    _slice(elem, slice(n - 1, None), axis),
                ],
                axis=axis,
            )
            for leaf, elem in zip(left, elems)
        ]
        odd_prefix = odd
    even_rest = combine(odd_prefix, even_sources)
    even = [
        mx.concatenate([_slice(elem, slice(None, 1), axis), rest], axis=axis)
        for elem, rest in zip(elems, even_rest)
    ]
    return [
        _interleave(even_leaf, odd_leaf, axis) for even_leaf, odd_leaf in zip(even, odd)
    ]


def _pairs(elems: Leaves, axis: int) -> tuple[Leaves, Leaves]:
    """Split each leaf into its even-indexed and odd-indexed elements.

    A trailing unpaired element is dropped here and re-added by the caller.
    """
    n = elems[0].shape[axis]
    return (
        [_slice(leaf, slice(None, n - n % 2, 2), axis) for leaf in elems],
        [_slice(leaf, slice(1, None, 2), axis) for leaf in elems],
    )


def _interleave(a: mx.array, b: mx.array, axis: int) -> mx.array:
    """Return ``a0 b0 a1 b1 ...``; ``a`` may hold one more element than ``b``."""
    axis %= a.ndim
    m = b.shape[axis]
    shape = list(a.shape)
    shape[axis] = 2 * m
    body = mx.stack([_slice(a, slice(None, m), axis), b], axis=axis + 1).reshape(shape)
    if a.shape[axis] == m:
        return body
    return mx.concatenate([body, _slice(a, slice(m, None), axis)], axis=axis)


def _slice(leaf: mx.array, selection: slice, axis: int) -> mx.array:
    indices = [slice(None)] * leaf.ndim
    indices[axis] = selection
    return leaf[tuple(indices)]


def _check_lengths(leaves: Leaves, axis: int) -> None:
    if not leaves:
        raise ValueError("associative_scan: elems has no arrays")
    if any(not -leaf.ndim <= axis < leaf.ndim for leaf in leaves):
        raise ValueError(f"associative_scan: axis {axis} is outside a leaf's shape")
    n = leaves[0].shape[axis]
    if any(leaf.shape[axis] != n for leaf in leaves):
        raise ValueError(
            "associative_scan: all arrays need the same scan length, saw "
            f"{[leaf.shape for leaf in leaves]}"
        )
