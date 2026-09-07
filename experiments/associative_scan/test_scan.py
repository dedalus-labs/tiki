"""Tests for the associative scan port. Each test names its invariant and witness."""

import importlib.util
import unittest
from collections.abc import Callable
from typing import Any

import tiki as tk
import numpy as np
from tiki.utils import tree_flatten, tree_unflatten

from scan import associative_scan

HAS_JAX = importlib.util.find_spec("jax") is not None


def strided_slice_autograd_is_fixed() -> bool:
    """True on a Tiki build with the normalize_slice fix; pip Tiki 0.32 lacks it."""
    grad = tk.vjp(lambda x: x[::2], (tk.array([1.0, 2.0]),), (tk.array([1.0]),))[1][0]
    return grad.tolist() == [1.0, 0.0]


FIXED = strided_slice_autograd_is_fixed()
LENGTHS = (0, 1, 2, 3, 4, 5, 7, 8, 16, 33, 129)


def sequential(fn: Callable[[Any, Any], Any], elems: Any) -> Any:
    """Reference fold along axis 0 over a pytree; the oracle for every test."""
    flat = tree_flatten(elems)
    keys = [key for key, _ in flat]
    leaves = [leaf for _, leaf in flat]
    length = leaves[0].shape[0]
    if length == 0:
        return elems
    at = lambda i: tree_unflatten([(key, leaf[i]) for key, leaf in zip(keys, leaves)])
    state, states = at(0), [at(0)]
    for i in range(1, length):
        state = fn(state, at(i))
        states.append(state)
    stacked = [
        tk.stack([[leaf for _, leaf in tree_flatten(state)][j] for state in states])
        for j in range(len(leaves))
    ]
    return tree_unflatten(list(zip(keys, stacked)))


def affine(
    left: tuple[tk.array, tk.array], right: tuple[tk.array, tk.array]
) -> tuple[tk.array, tk.array]:
    al, bl = left
    ar, br = right
    return ar * al, ar * bl + br


def rows(length: int, width: int, seed: int) -> tk.array:
    return tk.array(
        np.random.default_rng(seed).normal(size=(length, width)).astype(np.float32)
    )


def small_integer_matrices(length: int, seed: int) -> np.ndarray:
    """Entries in {-1, 0, 1} so every prefix product up to length 9 is exact in
    float32 on any backend, including TF32 matmul; the test then measures the
    scan, not the device's matmul precision."""
    return (
        np.random.default_rng(seed)
        .integers(-1, 2, size=(length, 2, 2))
        .astype(np.float32)
    )


def assert_close(actual: Any, expected: Any, name: str) -> None:
    actual_leaves = [leaf for _, leaf in tree_flatten(actual)]
    expected_leaves = [leaf for _, leaf in tree_flatten(expected)]
    tk.eval(actual_leaves, expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            atol=2e-5,
            rtol=2e-4,
            err_msg=name,
        )


class TestAssociativeScan(unittest.TestCase):
    # Invariant: scan with addition equals tk.cumsum on every length, both
    # directions, on a leading and a trailing axis.
    # Witness: random float32 rows for each length in LENGTHS.
    def test_cumsum_parity(self) -> None:
        for length in LENGTHS:
            x = rows(length, 3, length)
            for reverse in (False, True):
                got = associative_scan(tk.add, x, reverse=reverse)
                assert_close(
                    got,
                    tk.cumsum(x, axis=0, reverse=reverse),
                    f"{length=} reverse={reverse}",
                )
                got_t = associative_scan(tk.add, x.T, reverse=reverse, axis=1)
                assert_close(
                    got_t,
                    tk.cumsum(x.T, axis=1, reverse=reverse),
                    f"{length=} axis=1 reverse={reverse}",
                )

    # Invariant: combine order is preserved for a non-commutative operation.
    # Witness: prefix products of random 2x2 matrices versus a sequential fold.
    def test_noncommutative_matmul(self) -> None:
        for length in (1, 2, 3, 5, 8, 9):
            mats = tk.array(small_integer_matrices(length, length))
            assert_close(
                associative_scan(tk.matmul, mats),
                sequential(tk.matmul, mats),
                f"{length=}",
            )

    # Invariant: pytree inputs scan leaf-wise with one combine over the tree.
    # Witness: the affine pair (a, b) with a zero coefficient inside the row.
    def test_pytree_affine(self) -> None:
        for length in LENGTHS:
            a, b = rows(length, 3, length), rows(length, 3, length + 100)
            if length > 1:
                a[1, 0] = 0
            assert_close(
                associative_scan(affine, (a, b)),
                sequential(affine, (a, b)),
                f"{length=}",
            )

    # Invariant: reverse-mode derivatives of the tree match the sequential fold
    # for arbitrary cotangents on both outputs, with no registered backward.
    # Witness: the affine pair at lengths that cover both recursion parities.
    @unittest.skipUnless(FIXED, "requires the singleton slice derivative fix")
    def test_vjp_matches_sequential(self) -> None:
        for length in (1, 2, 3, 7, 16, 33):
            a, b, ga, gb = (rows(length, 3, length + k) for k in range(4))
            tree_grads = tk.vjp(
                lambda a, b: associative_scan(affine, (a, b)), (a, b), (ga, gb)
            )[1]
            seq_grads = tk.vjp(
                lambda a, b: sequential(affine, (a, b)), (a, b), (ga, gb)
            )[1]
            assert_close(tree_grads, seq_grads, f"{length=}")

    # Invariant: forward-mode derivatives match the sequential fold.
    # Witness: unit tangents on the affine pair at length 7 and 16.
    @unittest.skipUnless(FIXED, "requires the singleton slice derivative fix")
    def test_jvp_matches_sequential(self) -> None:
        for length in (7, 16):
            a, b = rows(length, 3, length), rows(length, 3, length + 1)
            tangents = (tk.ones_like(a), tk.ones_like(b))
            tree = tk.jvp(
                lambda a, b: associative_scan(affine, (a, b)), (a, b), tangents
            )[1]
            seq = tk.jvp(lambda a, b: sequential(affine, (a, b)), (a, b), tangents)[1]
            assert_close(tree, seq, f"{length=}")

    # Invariant: the scan composes with vmap the way JAX's does.
    # Witness: a batched cumsum equals cumsum along the scanned axis.
    @unittest.skipUnless(FIXED, "requires the singleton slice transformation fix")
    def test_vmap(self) -> None:
        x = tk.array(np.random.default_rng(0).normal(size=(4, 9, 2)).astype(np.float32))
        batched = tk.vmap(lambda row: associative_scan(tk.add, row))(x)
        assert_close(batched, tk.cumsum(x, axis=1), "vmap")

    # Invariant: the length-2 case, the one the Tiki slice fix repairs,
    # differentiates correctly through stride-2 pairing.
    # Witness: d(sum of scan)/dx for x of length 2 is [2, 1].
    @unittest.skipUnless(FIXED, "requires the singleton slice derivative fix")
    def test_length_two_vjp(self) -> None:
        x = tk.array([1.0, 2.0])
        grad = tk.grad(lambda x: tk.sum(associative_scan(tk.add, x)))(x)
        assert_close(grad, tk.array([2.0, 1.0]), "length-2 vjp")

    # Invariant: the VJP of x[::2] on a length-2 input routes the single
    # cotangent to element 0 only (the Tiki normalize_slice fix).
    # Witness: expected [1, 0]; unfixed Tiki 0.32 returns [1, 1].
    @unittest.skipUnless(FIXED, "requires the singleton slice derivative fix")
    def test_tiki_strided_slice_vjp_is_fixed(self) -> None:
        x = tk.array([1.0, 2.0])
        grad = tk.vjp(lambda x: x[::2], (x,), (tk.array([1.0]),))[1][0]
        assert_close(grad, tk.array([1.0, 0.0]), "x[::2] vjp")

    # Invariant: malformed inputs fail with a typed error, never a wrong result.
    # Witness: non-callable fn, an empty tree, and mismatched scan lengths.
    def test_errors(self) -> None:
        with self.assertRaises(TypeError):
            associative_scan(None, tk.zeros((3,)))
        with self.assertRaises(ValueError):
            associative_scan(tk.add, ())
        with self.assertRaises(ValueError):
            associative_scan(affine, (tk.zeros((3,)), tk.zeros((4,))))


@unittest.skipUnless(HAS_JAX, "needs jax")
class TestJaxParity(unittest.TestCase):
    """The port must match jax.lax.associative_scan within the stated tolerance."""

    def jax_scan(self, fn: Any, elems: Any, **kwargs: Any) -> Any:
        import jax

        return jax.lax.associative_scan(fn, elems, **kwargs)

    # Invariant: same results as JAX for add on every length and direction.
    # Witness: the same NumPy rows fed to both frameworks.
    def test_add(self) -> None:
        import jax.numpy as jnp

        for length in LENGTHS:
            x = (
                np.random.default_rng(length)
                .normal(size=(length, 3))
                .astype(np.float32)
            )
            for reverse in (False, True):
                ours = associative_scan(tk.add, tk.array(x), reverse=reverse)
                theirs = self.jax_scan(jnp.add, jnp.asarray(x), reverse=reverse)
                assert_close(
                    ours,
                    tk.array(np.asarray(theirs)),
                    f"{length=} reverse={reverse}",
                )

    # Invariant: same results as JAX for a non-commutative combine on axis 1.
    # Witness: prefix matrix products along the middle axis of a (2, length, 2, 2) array.
    def test_matmul_axis_one(self) -> None:
        import jax.numpy as jnp

        for length in (1, 2, 3, 6, 9):
            x = np.stack(
                [
                    small_integer_matrices(length, length),
                    small_integer_matrices(length, length + 50),
                ]
            )
            ours = associative_scan(tk.matmul, tk.array(x), axis=1)
            theirs = self.jax_scan(jnp.matmul, jnp.asarray(x), axis=1)
            assert_close(ours, tk.array(np.asarray(theirs)), f"{length=}")

    # Invariant: same results as JAX for a tuple pytree combine.
    # Witness: the affine pair, matching JAX's own tuple-of-arrays example shape.
    def test_pytree(self) -> None:
        import jax.numpy as jnp

        def jax_affine(left: Any, right: Any) -> Any:
            al, bl = left
            ar, br = right
            return ar * al, ar * bl + br

        for length in (1, 2, 5, 16, 33):
            a = (
                np.random.default_rng(length)
                .normal(size=(length, 3))
                .astype(np.float32)
            )
            b = (
                np.random.default_rng(length + 1)
                .normal(size=(length, 3))
                .astype(np.float32)
            )
            ours = associative_scan(affine, (tk.array(a), tk.array(b)))
            theirs = self.jax_scan(jax_affine, (jnp.asarray(a), jnp.asarray(b)))
            assert_close(
                ours,
                tuple(tk.array(np.asarray(leaf)) for leaf in theirs),
                f"{length=}",
            )


if __name__ == "__main__":
    unittest.main()
