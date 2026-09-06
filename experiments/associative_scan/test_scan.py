"""Tests for the associative scan port. Each test names its invariant and witness."""

import importlib.util
import unittest
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from scan import associative_scan

HAS_JAX = importlib.util.find_spec("jax") is not None


def strided_slice_autograd_is_fixed() -> bool:
    """True on a Tiki build with the normalize_slice fix; pip MLX 0.32 lacks it."""
    grad = mx.vjp(lambda x: x[::2], (mx.array([1.0, 2.0]),), (mx.array([1.0]),))[1][0]
    return grad.tolist() == [1.0, 0.0]


FIXED = strided_slice_autograd_is_fixed()
LENGTHS = (0, 1, 2, 3, 4, 5, 7, 8, 16, 33, 129)


def sequential(fn: Callable[[Any, Any], Any], elems: Any) -> Any:
    """Reference fold along axis 0 over a pytree; the oracle for every test."""
    flat = tree_flatten(elems)
    keys = [key for key, _ in flat]
    leaves = [leaf for _, leaf in flat]
    n = leaves[0].shape[0]
    if n == 0:
        return elems
    at = lambda i: tree_unflatten([(key, leaf[i]) for key, leaf in zip(keys, leaves)])
    state, states = at(0), [at(0)]
    for i in range(1, n):
        state = fn(state, at(i))
        states.append(state)
    stacked = [
        mx.stack([[leaf for _, leaf in tree_flatten(state)][j] for state in states])
        for j in range(len(leaves))
    ]
    return tree_unflatten(list(zip(keys, stacked)))


def affine(
    left: tuple[mx.array, mx.array], right: tuple[mx.array, mx.array]
) -> tuple[mx.array, mx.array]:
    al, bl = left
    ar, br = right
    return ar * al, ar * bl + br


def rows(n: int, width: int, seed: int) -> mx.array:
    return mx.array(
        np.random.default_rng(seed).normal(size=(n, width)).astype(np.float32)
    )


def small_integer_matrices(n: int, seed: int) -> np.ndarray:
    """Entries in {-1, 0, 1} so every prefix product up to length 9 is exact in
    float32 on any backend, including TF32 matmul; the test then measures the
    scan, not the device's matmul precision."""
    return (
        np.random.default_rng(seed).integers(-1, 2, size=(n, 2, 2)).astype(np.float32)
    )


def assert_close(actual: Any, expected: Any, name: str) -> None:
    actual_leaves = [leaf for _, leaf in tree_flatten(actual)]
    expected_leaves = [leaf for _, leaf in tree_flatten(expected)]
    mx.eval(actual_leaves, expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            atol=2e-5,
            rtol=2e-4,
            err_msg=name,
        )


@unittest.skipUnless(
    FIXED, "MLX build lacks the normalize_slice fix; strided-slice VJP is wrong"
)
class TestAssociativeScan(unittest.TestCase):
    # Invariant: scan with addition equals mx.cumsum on every length, both
    # directions, on a leading and a trailing axis.
    # Witness: random float32 rows for each length in LENGTHS.
    def test_cumsum_parity(self) -> None:
        for n in LENGTHS:
            x = rows(n, 3, n)
            for reverse in (False, True):
                got = associative_scan(mx.add, x, reverse=reverse)
                assert_close(
                    got,
                    mx.cumsum(x, axis=0, reverse=reverse),
                    f"n={n} reverse={reverse}",
                )
                got_t = associative_scan(mx.add, x.T, reverse=reverse, axis=1)
                assert_close(
                    got_t,
                    mx.cumsum(x.T, axis=1, reverse=reverse),
                    f"n={n} axis=1 reverse={reverse}",
                )

    # Invariant: combine order is preserved for a non-commutative operation.
    # Witness: prefix products of random 2x2 matrices versus a sequential fold.
    def test_noncommutative_matmul(self) -> None:
        for n in (1, 2, 3, 5, 8, 9):
            mats = mx.array(small_integer_matrices(n, n))
            assert_close(
                associative_scan(mx.matmul, mats), sequential(mx.matmul, mats), f"n={n}"
            )

    # Invariant: pytree inputs scan leaf-wise with one combine over the tree.
    # Witness: the affine pair (a, b) with a zero coefficient inside the row.
    def test_pytree_affine(self) -> None:
        for n in LENGTHS:
            a, b = rows(n, 3, n), rows(n, 3, n + 100)
            if n > 1:
                a[1, 0] = 0
            assert_close(
                associative_scan(affine, (a, b)), sequential(affine, (a, b)), f"n={n}"
            )

    # Invariant: reverse-mode derivatives of the tree match the sequential fold
    # for arbitrary cotangents on both outputs, with no registered backward.
    # Witness: the affine pair at lengths that cover both recursion parities.
    def test_vjp_matches_sequential(self) -> None:
        for n in (1, 2, 3, 7, 16, 33):
            a, b, ga, gb = (rows(n, 3, n + k) for k in range(4))
            tree_grads = mx.vjp(
                lambda a, b: associative_scan(affine, (a, b)), (a, b), (ga, gb)
            )[1]
            seq_grads = mx.vjp(
                lambda a, b: sequential(affine, (a, b)), (a, b), (ga, gb)
            )[1]
            assert_close(tree_grads, seq_grads, f"n={n}")

    # Invariant: forward-mode derivatives match the sequential fold.
    # Witness: unit tangents on the affine pair at length 7 and 16.
    def test_jvp_matches_sequential(self) -> None:
        for n in (7, 16):
            a, b = rows(n, 3, n), rows(n, 3, n + 1)
            tangents = (mx.ones_like(a), mx.ones_like(b))
            tree = mx.jvp(
                lambda a, b: associative_scan(affine, (a, b)), (a, b), tangents
            )[1]
            seq = mx.jvp(lambda a, b: sequential(affine, (a, b)), (a, b), tangents)[1]
            assert_close(tree, seq, f"n={n}")

    # Invariant: the scan composes with vmap the way JAX's does.
    # Witness: a batched cumsum equals cumsum along the scanned axis.
    def test_vmap(self) -> None:
        x = mx.array(np.random.default_rng(0).normal(size=(4, 9, 2)).astype(np.float32))
        batched = mx.vmap(lambda row: associative_scan(mx.add, row))(x)
        assert_close(batched, mx.cumsum(x, axis=1), "vmap")

    # Invariant: the length-2 case, the one the MLX slice fix repairs,
    # differentiates correctly through stride-2 pairing.
    # Witness: d(sum of scan)/dx for x of length 2 is [2, 1].
    def test_length_two_vjp(self) -> None:
        x = mx.array([1.0, 2.0])
        grad = mx.grad(lambda x: mx.sum(associative_scan(mx.add, x)))(x)
        assert_close(grad, mx.array([2.0, 1.0]), "length-2 vjp")

    # Invariant: the VJP of x[::2] on a length-2 input routes the single
    # cotangent to element 0 only (the Tiki normalize_slice fix).
    # Witness: expected [1, 0]; unfixed MLX 0.32 returns [1, 1].
    def test_mlx_strided_slice_vjp_is_fixed(self) -> None:
        x = mx.array([1.0, 2.0])
        grad = mx.vjp(lambda x: x[::2], (x,), (mx.array([1.0]),))[1][0]
        assert_close(grad, mx.array([1.0, 0.0]), "x[::2] vjp")

    # Invariant: malformed inputs fail with a typed error, never a wrong result.
    # Witness: non-callable fn, an empty tree, and mismatched scan lengths.
    def test_errors(self) -> None:
        with self.assertRaises(TypeError):
            associative_scan(None, mx.zeros((3,)))
        with self.assertRaises(ValueError):
            associative_scan(mx.add, ())
        with self.assertRaises(ValueError):
            associative_scan(affine, (mx.zeros((3,)), mx.zeros((4,))))


@unittest.skipUnless(HAS_JAX and FIXED, "needs jax and the normalize_slice fix")
class TestJaxParity(unittest.TestCase):
    """The port must reproduce jax.lax.associative_scan bit-for-bit up to float32 rounding."""

    def jax_scan(self, fn: Any, elems: Any, **kwargs: Any) -> Any:
        import jax

        return jax.lax.associative_scan(fn, elems, **kwargs)

    # Invariant: same results as JAX for add on every length and direction.
    # Witness: the same NumPy rows fed to both frameworks.
    def test_add(self) -> None:
        import jax.numpy as jnp

        for n in LENGTHS:
            x = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32)
            for reverse in (False, True):
                ours = associative_scan(mx.add, mx.array(x), reverse=reverse)
                theirs = self.jax_scan(jnp.add, jnp.asarray(x), reverse=reverse)
                assert_close(
                    ours, mx.array(np.asarray(theirs)), f"n={n} reverse={reverse}"
                )

    # Invariant: same results as JAX for a non-commutative combine on axis 1.
    # Witness: prefix matrix products along the middle axis of a (2, n, 2, 2) array.
    def test_matmul_axis_one(self) -> None:
        import jax.numpy as jnp

        for n in (1, 2, 3, 6, 9):
            x = np.stack(
                [small_integer_matrices(n, n), small_integer_matrices(n, n + 50)]
            )
            ours = associative_scan(mx.matmul, mx.array(x), axis=1)
            theirs = self.jax_scan(jnp.matmul, jnp.asarray(x), axis=1)
            assert_close(ours, mx.array(np.asarray(theirs)), f"n={n}")

    # Invariant: same results as JAX for a tuple pytree combine.
    # Witness: the affine pair, matching JAX's own tuple-of-arrays example shape.
    def test_pytree(self) -> None:
        import jax.numpy as jnp

        def jax_affine(left: Any, right: Any) -> Any:
            al, bl = left
            ar, br = right
            return ar * al, ar * bl + br

        for n in (1, 2, 5, 16, 33):
            a = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32)
            b = np.random.default_rng(n + 1).normal(size=(n, 3)).astype(np.float32)
            ours = associative_scan(affine, (mx.array(a), mx.array(b)))
            theirs = self.jax_scan(jax_affine, (jnp.asarray(a), jnp.asarray(b)))
            assert_close(
                ours, tuple(mx.array(np.asarray(leaf)) for leaf in theirs), f"n={n}"
            )


if __name__ == "__main__":
    unittest.main()
