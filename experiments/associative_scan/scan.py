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
consistent stop and stride), so ``Slice::vjp`` and ``Slice::jvp`` are correct;
``test_scan`` skips on an MLX build without it.

[BLE1990] Blelloch, Guy E. 1990. "Prefix Sums and Their Applications."
CMU-CS-90-190.
"""

from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

Leaves = list[mx.array]


def associative_scan(
    fn: Callable[[Any, Any], Any],
    elems: Any,
    reverse: bool = False,
    axis: int = 0,
) -> Any:
    """Scan ``elems`` along ``axis`` with the associative binary operation ``fn``.

    ``fn(a, b)`` receives and returns pytrees shaped like ``elems`` and must be
    associative. Element ``k`` of the result is ``fn`` folded over the first
    ``k + 1`` elements; with ``reverse`` it is folded over the last ones. The
    combine order is preserved, so non-commutative operations are correct.
    """
    if not callable(fn):
        raise TypeError("associative_scan: fn must be callable")
    flat = tree_flatten(elems)
    keys = [key for key, _ in flat]
    leaves = [_to_front(leaf, axis) for _, leaf in flat]
    _check_lengths(leaves)
    if reverse:
        leaves = [leaf[::-1] for leaf in leaves]

    def combine(a: Leaves, b: Leaves) -> Leaves:
        c = fn(tree_unflatten(list(zip(keys, a))), tree_unflatten(list(zip(keys, b))))
        return [leaf for _, leaf in tree_flatten(c)]

    scans = _scan(combine, leaves)
    if reverse:
        scans = [scan[::-1] for scan in scans]
    scans = [_from_front(scan, axis) for scan in scans]
    return tree_unflatten(list(zip(keys, scans)))


def _scan(combine: Callable[[Leaves, Leaves], Leaves], elems: Leaves) -> Leaves:
    n = elems[0].shape[0]
    if n < 2:
        return elems
    left, right = _pairs(elems)
    odd = _scan(combine, combine(left, right))
    if n % 2 == 0:
        even_sources = [leaf[1:] for leaf in left]
        odd_prefix = [scan[:-1] for scan in odd]
    else:
        even_sources = [
            mx.concatenate([leaf[1:], elem[n - 1 :]]) for leaf, elem in zip(left, elems)
        ]
        odd_prefix = odd
    even_rest = combine(odd_prefix, even_sources)
    even = [mx.concatenate([elem[:1], rest]) for elem, rest in zip(elems, even_rest)]
    return [_interleave(even_leaf, odd_leaf) for even_leaf, odd_leaf in zip(even, odd)]


def _pairs(elems: Leaves) -> tuple[Leaves, Leaves]:
    """Split each leaf into its even-indexed and odd-indexed elements.

    A trailing unpaired element is dropped here and re-added by the caller.
    """
    n = elems[0].shape[0]
    return [leaf[: n - n % 2 : 2] for leaf in elems], [leaf[1::2] for leaf in elems]


def _interleave(a: mx.array, b: mx.array) -> mx.array:
    """Return ``a0 b0 a1 b1 ...``; ``a`` may hold one more element than ``b``."""
    m = b.shape[0]
    body = mx.stack([a[:m], b], axis=1).reshape(2 * m, *a.shape[1:])
    if a.shape[0] == m:
        return body
    return mx.concatenate([body, a[m:]])


def _to_front(leaf: mx.array, axis: int) -> mx.array:
    return leaf if axis == 0 else mx.moveaxis(leaf, axis, 0)


def _from_front(leaf: mx.array, axis: int) -> mx.array:
    return leaf if axis == 0 else mx.moveaxis(leaf, 0, axis)


def _check_lengths(leaves: Leaves) -> None:
    if not leaves:
        raise ValueError("associative_scan: elems has no arrays")
    n = leaves[0].shape[0]
    if any(leaf.shape[0] != n for leaf in leaves):
        raise ValueError(
            "associative_scan: all arrays need the same scan length, saw "
            f"{[leaf.shape for leaf in leaves]}"
        )
