"""Tests for the CUDA affine scan. The oracle is the generic tree and its autodiff."""

import unittest

import numpy as np
import tiki as tk

from scan import associative_scan

HAS_CUDA = tk.cuda.is_available()
if HAS_CUDA:
    from affine_scan import ScanContractError, affine_scan

Pair = tuple[tk.array, tk.array]
LENGTHS = (1, 2, 3, 7, 31, 129, 1024, 2048)


def affine(left: Pair, right: Pair) -> Pair:
    al, bl = left
    ar, br = right
    return ar * al, ar * bl + br


def tree_scan(a: tk.array, b: tk.array) -> Pair:
    return associative_scan(affine, (a, b), axis=1)


def rows(batch: int, time: int, seed: int) -> tk.array:
    return tk.array(
        np.random.default_rng(seed).normal(size=(batch, time)).astype(np.float32)
    )


def assert_close(actual: Pair, expected: Pair, name: str) -> None:
    tk.eval(actual, expected)
    tk.synchronize()
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
                a = tk.ones((1, time))
                a[0, 0] = coefficient
                b = tk.arange(1, time + 1, dtype=tk.float32)[None]
                _, offsets = affine_scan(a, b)
                tk.eval(offsets)
                tk.synchronize()
                np.testing.assert_array_equal(
                    np.asarray(offsets), np.asarray(tk.cumsum(b, axis=1))
                )

    def test_first_coefficient_has_no_offset_cotangent(self) -> None:
        a = tk.ones((1, 3))
        b = tk.ones_like(a)
        gp = tk.zeros_like(a)
        gh = tk.array([[float("inf"), 0.0, 0.0]])
        da, _ = tk.vjp(affine_scan, (a, b), (gp, gh))[1]
        tk.eval(da)
        tk.synchronize()
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

    # Invariant: the registered VJP equals Tiki's differentiation of the
    # generic tree for arbitrary cotangents on both outputs (JAX's method for
    # validating a kernel gradient against the tree).
    # Witness: random cotangents at every contract length.
    def test_vjp_matches_tree_autodiff(self) -> None:
        for time in LENGTHS:
            a, b, gp, gh = (rows(5, time, time + k) for k in range(4))
            a[0, time // 2] = 0.0
            kernel_grads = tk.vjp(affine_scan, (a, b), (gp, gh))[1]
            tree_grads = tk.vjp(tree_scan, (a, b), (gp, gh))[1]
            assert_close(kernel_grads, tree_grads, f"time={time}")

    # Invariant: the registered VJP survives tk.compile.
    # Witness: a compiled VJP at time 129 against the tree.
    def test_compiled_vjp(self) -> None:
        a, b, gp, gh = (rows(3, 129, 10 + k) for k in range(4))
        compiled = tk.compile(
            lambda a, b, gp, gh: tk.vjp(affine_scan, (a, b), (gp, gh))[1]
        )
        assert_close(
            compiled(a, b, gp, gh),
            tk.vjp(tree_scan, (a, b), (gp, gh))[1],
            "compiled vjp",
        )

    # Invariant: inputs outside the contract raise ScanContractError, never a
    # silent fallback to the tree.
    # Witness: time 0, time 2049, a 1-D input, a float16 input, and a shape mismatch.
    def test_contract(self) -> None:
        good = rows(2, 8, 0)
        for a, b in [
            (tk.zeros((2, 0)), tk.zeros((2, 0))),
            (tk.zeros((2, 2049)), tk.zeros((2, 2049))),
            (tk.zeros((8,)), tk.zeros((8,))),
            (good.astype(tk.float16), good.astype(tk.float16)),
            (good, rows(2, 9, 1)),
        ]:
            with self.assertRaises(ScanContractError):
                affine_scan(a, b)


if __name__ == "__main__":
    unittest.main()
