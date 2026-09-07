"""Tests for the CUDA affine scan. The oracle is the generic tree and its autodiff."""

import unittest

import mlx.core as mx
import numpy as np

from scan import associative_scan

HAS_CUDA = mx.cuda.is_available()
if HAS_CUDA:
    from affine_scan import ScanContractError, affine_scan

Pair = tuple[mx.array, mx.array]
LENGTHS = (1, 2, 3, 7, 31, 129, 1024, 2048)


def affine(left: Pair, right: Pair) -> Pair:
    al, bl = left
    ar, br = right
    return ar * al, ar * bl + br


def tree_scan(a: mx.array, b: mx.array) -> Pair:
    return associative_scan(affine, (a, b), axis=1)


def rows(batch: int, time: int, seed: int) -> mx.array:
    return mx.array(
        np.random.default_rng(seed).normal(size=(batch, time)).astype(np.float32)
    )


def assert_close(actual: Pair, expected: Pair, name: str) -> None:
    mx.eval(actual, expected)
    mx.synchronize()
    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            atol=3e-5,
            rtol=3e-4,
            err_msg=name,
        )


@unittest.skipUnless(HAS_CUDA, "the affine scan kernel needs CUDA")
class TestAffineScan(unittest.TestCase):
    def test_first_offset_does_not_depend_on_the_first_coefficient(self) -> None:
        for time in (1, 3, 8):
            for coefficient in (float("inf"), -float("inf"), float("nan")):
                a = mx.ones((1, time))
                a[0, 0] = coefficient
                b = mx.arange(1, time + 1, dtype=mx.float32)[None]
                _, offsets = affine_scan(a, b)
                mx.eval(offsets)
                mx.synchronize()
                np.testing.assert_array_equal(
                    np.asarray(offsets), np.asarray(mx.cumsum(b, axis=1))
                )

    def test_first_coefficient_has_no_offset_cotangent(self) -> None:
        a = mx.ones((1, 3))
        b = mx.ones_like(a)
        gp = mx.zeros_like(a)
        gh = mx.array([[float("inf"), 0.0, 0.0]])
        da, _ = mx.vjp(affine_scan, (a, b), (gp, gh))[1]
        mx.eval(da)
        mx.synchronize()
        self.assertEqual(da[0, 0].item(), 0.0)

    # Invariant: the kernel's forward equals the generic tree at every
    # contract length, including a zero coefficient mid-row.
    # Witness: batch 5 rows for each length in LENGTHS.
    def test_forward_matches_tree(self) -> None:
        for time in LENGTHS:
            a = rows(5, time, time)
            a[0, time // 2] = 0.0
            b = rows(5, time, time + 1)
            assert_close(affine_scan(a, b), tree_scan(a, b), f"time={time}")

    # Invariant: the registered VJP equals MLX's differentiation of the
    # generic tree for arbitrary cotangents on both outputs (JAX's method for
    # validating a kernel gradient against the tree).
    # Witness: random cotangents at every contract length.
    def test_vjp_matches_tree_autodiff(self) -> None:
        for time in LENGTHS:
            a, b, gp, gh = (rows(5, time, time + k) for k in range(4))
            a[0, time // 2] = 0.0
            kernel_grads = mx.vjp(affine_scan, (a, b), (gp, gh))[1]
            tree_grads = mx.vjp(tree_scan, (a, b), (gp, gh))[1]
            assert_close(kernel_grads, tree_grads, f"time={time}")

    # Invariant: the registered VJP survives mx.compile.
    # Witness: a compiled VJP at time 129 against the tree.
    def test_compiled_vjp(self) -> None:
        a, b, gp, gh = (rows(3, 129, 10 + k) for k in range(4))
        compiled = mx.compile(
            lambda a, b, gp, gh: mx.vjp(affine_scan, (a, b), (gp, gh))[1]
        )
        assert_close(
            compiled(a, b, gp, gh),
            mx.vjp(tree_scan, (a, b), (gp, gh))[1],
            "compiled vjp",
        )

    # Invariant: inputs outside the contract raise ScanContractError, never a
    # silent fallback to the tree.
    # Witness: time 0, time 2049, a 1-D input, a float16 input, and a shape mismatch.
    def test_contract(self) -> None:
        good = rows(2, 8, 0)
        for a, b in [
            (mx.zeros((2, 0)), mx.zeros((2, 0))),
            (mx.zeros((2, 2049)), mx.zeros((2, 2049))),
            (mx.zeros((8,)), mx.zeros((8,))),
            (good.astype(mx.float16), good.astype(mx.float16)),
            (good, rows(2, 9, 1)),
        ]:
            with self.assertRaises(ScanContractError):
                affine_scan(a, b)


if __name__ == "__main__":
    unittest.main()
