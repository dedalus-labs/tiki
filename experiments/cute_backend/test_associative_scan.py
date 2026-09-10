"""Test scan lowering and execution on the target selected by TIKI_TEST_ARCH.

The execution oracle is the generic Blelloch tree in ``experiments/associative_scan``
differentiated by MLX, the way JAX checks its native cumsum gradient against
``associative_scan``. Lengths straddle every level of the hierarchy: one thread's
chunk, one warp, one block, one tile, and several levels of tile recursion.
Affine derivatives use independent sequential float64 oracles.
"""

import os
import sys
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np

from associative_scan import (
    ArrayTree,
    ScanContractError,
    associative_scan,
    flatten,
    operation,
)
from test_scan_reference import affine_jvp, affine_vjp, complex_jvp, complex_vjp
from tiki_compiler.graph import Profile, Shape, UnsupportedGraphError
from tiki_compiler.scan import lower_apply, lower_tile_scan
from tiki_compiler.scan_schedule import ScanSchedule

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "associative_scan"))
from scan import associative_scan as tree_scan

ARCH = os.environ.get("TIKI_TEST_ARCH", "sm_90")
SMALL = ScanSchedule(arch=ARCH, threads=32, elements_per_thread=1)
MEDIUM = ScanSchedule(arch=ARCH, threads=64, elements_per_thread=2)
LARGE = ScanSchedule(arch=ARCH, threads=128, elements_per_thread=4)
LENGTHS = (1, 2, 3, 5, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1025, 4097)

ON_DEVICE = mx.cuda.is_available() and mx.device_info(mx.gpu)["architecture"] == ARCH
device = unittest.skipUnless(ON_DEVICE, f"requires MLX CUDA on {ARCH}")


def affine(left: Sequence[mx.array], right: Sequence[mx.array]) -> tuple[mx.array, mx.array]:
    """The affine recurrence h_t = a_t h_{t-1} + b_t; non-commutative."""
    a_left, b_left = left
    a_right, b_right = right
    return (a_right * a_left, a_right * b_left + b_right)


def random(shape: Shape, seed: int, scale: float = 1.0) -> mx.array:
    return mx.array(np.random.default_rng(seed).standard_normal(shape).astype(np.float32) * scale)


def close(a: mx.array, b: mx.array, tolerance: float = 1e-4) -> bool:
    return bool(np.allclose(np.array(a), np.array(b), rtol=tolerance, atol=tolerance))


def flat_arrays(tree: ArrayTree) -> list[mx.array]:
    return [leaf for _, leaf in flatten(tree)]


class LoweringTest(unittest.TestCase):
    def test_combine_graph_is_scalar_and_elementwise(self) -> None:
        op = operation(affine, ("0", "1"), 1, SMALL)
        self.assertEqual(len(op.graph.inputs), 4)
        self.assertEqual(len(op.graph.outputs), 2)
        self.assertTrue(all(value.shape == () for value in op.graph.inputs))

    def test_tile_kernel_addresses_through_axis_layouts(self) -> None:
        op = operation(mx.add, ("",), 0, LARGE)
        lowered = lower_tile_scan(op.graph, (Profile(shape=(513, 4), strides=(1, 513)),), 0, LARGE)
        self.assertIn('"((1,4),513):((0,513),1)"', lowered.mlir)
        self.assertIn('"((1,4),513):((0,1),4)"', lowered.mlir)
        self.assertIn('"(4,2):(2,1)"', lowered.mlir)
        self.assertEqual(lowered.grid, (2 * 4 * 128, 1, 1))
        self.assertEqual(lowered.output_shapes, ((513, 4), (4, 2)))
        self.assertEqual(lowered.mlir.count("nvvm.shfl.sync up"), 6)
        self.assertEqual(lowered.shared_memory_bytes, 16)

    def test_single_warp_needs_no_shared_memory(self) -> None:
        op = operation(mx.add, ("",), 0, SMALL)
        lowered = lower_tile_scan(op.graph, (Profile(shape=(7,), strides=(1,)),), 0, SMALL)
        self.assertNotIn("smem", lowered.mlir)
        self.assertEqual(lowered.shared_memory_bytes, 0)

    def test_apply_kernel_folds_the_previous_tile(self) -> None:
        op = operation(affine, ("0", "1"), 1, MEDIUM)
        lowered = lower_apply(op.graph, (3, 300), 1, 3, MEDIUM)
        self.assertEqual(
            len(
                [
                    line
                    for line in lowered.mlir.splitlines()
                    if "%arg" in line and "cuda.kernel" in line
                ]
            ),
            1,
        )
        self.assertIn("%has_prefix = arith.cmpi uge, %tile, %one", lowered.mlir)
        self.assertEqual(lowered.output_shapes, ((3, 300), (3, 300)))

    def test_contract(self) -> None:
        with self.assertRaises(ScanContractError):
            associative_scan(mx.add, [])
        with self.assertRaises(ScanContractError):
            associative_scan(mx.add, mx.zeros((3, 4)), axis=2)
        op = operation(mx.add, ("",), 0, SMALL)
        with self.assertRaises(ScanContractError):
            op.check((mx.zeros((3,), dtype=mx.float16),))


@device
class ForwardTest(unittest.TestCase):
    def test_cumsum_every_length_and_schedule(self) -> None:
        for schedule in (SMALL, MEDIUM, LARGE):
            for length in LENGTHS:
                x = random((3, length), length)
                result = associative_scan(mx.add, x, axis=1, schedule=schedule)
                self.assertTrue(close(result, mx.cumsum(x, axis=1), 1e-3), (schedule, length))

    def test_reverse_matches_reverse_cumsum(self) -> None:
        for length in (1, 2, 33, 513, 4097):
            x = random((2, length), length)
            result = associative_scan(mx.add, x, axis=1, reverse=True, schedule=MEDIUM)
            self.assertTrue(close(result, mx.cumsum(x, axis=1, reverse=True), 1e-3), length)

    def test_any_axis_of_a_rank_three_array(self) -> None:
        x = random((6, 70, 5), 7)
        for axis in (0, 1, 2, -1):
            result = associative_scan(mx.add, x, axis=axis, schedule=SMALL)
            self.assertTrue(close(result, mx.cumsum(x, axis=axis), 1e-3), axis)

    def test_affine_matches_the_tree(self) -> None:
        for length in (1, 2, 3, 64, 65, 1000, 4097):
            a, b = random((4, length), length, 0.9), random((4, length), length + 1)
            result = associative_scan(affine, (a, b), axis=1, schedule=MEDIUM)
            expected = tree_scan(affine, (a, b), axis=1)
            for got, want in zip(flat_arrays(result), flat_arrays(expected), strict=True):
                self.assertTrue(close(got, want, 1e-3), length)

    def test_pytree_leaves(self) -> None:
        x = random((5, 100), 3)

        def combine(left: dict[str, mx.array], right: dict[str, mx.array]) -> dict[str, mx.array]:
            return {
                "sum": left["sum"] + right["sum"],
                "max": mx.maximum(left["max"], right["max"]),
            }

        with self.assertRaisesRegex(UnsupportedGraphError, "Maximum"):
            associative_scan(combine, {"sum": x, "max": x}, axis=1, schedule=SMALL)

    def test_strided_views_are_consumed_in_place(self) -> None:
        base = random((300, 7), 5)
        transposed = base.T
        sliced = base[3:, 2:6]
        for x, axis in ((transposed, 1), (sliced, 0), (base[::-1], 0)):
            result = associative_scan(mx.add, x, axis=axis, schedule=MEDIUM)
            self.assertTrue(close(result, mx.cumsum(x, axis=axis), 1e-3))

    def test_empty(self) -> None:
        x = mx.zeros((0, 4))
        result = associative_scan(mx.add, x, axis=1)
        self.assertEqual(result.shape, (0, 4))


@device
class DerivativeTest(unittest.TestCase):
    def check_affine_derivatives(self, elems: tuple[mx.array, ...], schedule: ScanSchedule) -> None:
        def compiled(*leaves: mx.array) -> list[mx.array]:
            result = flat_arrays(associative_scan(affine, leaves, axis=1, schedule=schedule))
            return result

        for transform, reference, seed in ((mx.vjp, affine_vjp, 100), (mx.jvp, affine_jvp, 200)):
            directions = [random(leaf.shape, seed + i) for i, leaf in enumerate(elems)]
            derivatives = transform(compiled, list(elems), directions)[1]
            for actual, expected in zip(derivatives, reference(elems, directions), strict=True):
                np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4, atol=1e-4)

    def test_cumsum_gradient_is_reverse_cumsum(self) -> None:
        for length in (1, 2, 33, 513, 4097):
            x = random((3, length), length)
            grad = mx.grad(lambda v: associative_scan(mx.add, v, axis=1, schedule=MEDIUM).sum())(x)
            self.assertTrue(
                close(grad, mx.cumsum(mx.ones_like(x), axis=1, reverse=True), 1e-3),
                length,
            )

    def test_affine_derivatives_match_sequential_recurrence(self) -> None:
        for length in (1, 2, 3, 64, 65, 300, 4097):
            a, b = random((3, length), length, 0.9), random((3, length), length + 1)
            self.check_affine_derivatives((a, b), MEDIUM)

    def test_complex_affine_couples_every_leaf(self) -> None:
        """``z_t = w_t z_{t-1} + v_t`` over complex ``w`` and ``v`` as four real
        leaves: associative, non-commutative, and every Jacobian block is dense."""

        def complex_affine(
            left: Sequence[mx.array], right: Sequence[mx.array]
        ) -> tuple[mx.array, ...]:
            wr_l, wi_l, vr_l, vi_l = left
            wr_r, wi_r, vr_r, vi_r = right
            return (
                wr_r * wr_l - wi_r * wi_l,
                wr_r * wi_l + wi_r * wr_l,
                wr_r * vr_l - wi_r * vi_l + vr_r,
                wr_r * vi_l + wi_r * vr_l + vi_r,
            )

        x = tuple(random((2, 200), 10 + leaf, 0.6) for leaf in range(4))
        result = associative_scan(complex_affine, x, axis=1, schedule=SMALL)
        for got, want in zip(
            flat_arrays(result), flat_arrays(tree_scan(complex_affine, x, axis=1)), strict=True
        ):
            self.assertTrue(close(got, want, 1e-3))

        def compiled(*leaves: mx.array) -> list[mx.array]:
            result = flat_arrays(associative_scan(complex_affine, leaves, axis=1, schedule=SMALL))
            return result

        for transform, reference, seed in ((mx.vjp, complex_vjp, 100), (mx.jvp, complex_jvp, 200)):
            directions = [random((2, 200), seed + leaf) for leaf in range(4)]
            derivatives = transform(compiled, list(x), directions)[1]
            for actual, expected in zip(derivatives, reference(x, directions), strict=True):
                np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4, atol=1e-4)

    def test_reverse_derivatives(self) -> None:
        a, b = random((2, 300), 1, 0.9), random((2, 300), 2)

        def compiled(a: mx.array, b: mx.array) -> list[mx.array]:
            return flat_arrays(
                associative_scan(affine, (a, b), axis=1, reverse=True, schedule=MEDIUM)
            )

        def tree(a: mx.array, b: mx.array) -> list[mx.array]:
            return flat_arrays(tree_scan(affine, (a, b), axis=1, reverse=True))

        cotangents = [random((2, 300), 3), random((2, 300), 4)]
        got = mx.vjp(compiled, [a, b], cotangents)[1]
        want = mx.vjp(tree, [a, b], cotangents)[1]
        for actual, expected in zip(got, want, strict=True):
            self.assertTrue(close(actual, expected, 1e-3))

    def test_derivatives_through_strided_views(self) -> None:
        base = random((300, 3), 9, 0.9)
        other = random((300, 3), 10)

        def compiled(base: mx.array, other: mx.array) -> list[mx.array]:
            return flat_arrays(associative_scan(affine, (base.T, other.T), axis=1, schedule=MEDIUM))

        def tree(base: mx.array, other: mx.array) -> list[mx.array]:
            return flat_arrays(tree_scan(affine, (base.T, other.T), axis=1))

        cotangents = [random((3, 300), 11), random((3, 300), 12)]
        got = mx.vjp(compiled, [base, other], cotangents)[1]
        want = mx.vjp(tree, [base, other], cotangents)[1]
        for actual, expected in zip(got, want, strict=True):
            self.assertTrue(close(actual, expected, 1e-3))

    def test_training_a_linear_recurrence(self) -> None:
        """A diagonal linear SSM trained with value_and_grad; the first step's
        gradient equals the tree's and the loss falls over the run."""
        batch, length, width = 4, 700, 8
        inputs = random((batch, length, width), 20)
        teacher = {"decay": random((width,), 21, 0.5), "gain": random((width,), 22)}

        def model(
            params: dict[str, mx.array], inputs: mx.array, scan: Callable[..., ArrayTree]
        ) -> mx.array:
            decay = mx.sigmoid(params["decay"]) * mx.ones_like(inputs)
            drive = inputs * params["gain"]
            _, hidden = flat_arrays(scan(affine, (decay, drive), axis=1))
            return hidden

        target = model(teacher, inputs, tree_scan)
        params = {"decay": mx.zeros((width,)), "gain": mx.ones((width,))}

        def loss(params: dict[str, mx.array], scan: Callable[..., ArrayTree]) -> mx.array:
            return mx.mean((model(params, inputs, scan) - target) ** 2)

        def compiled(params: dict[str, mx.array]) -> mx.array:
            return loss(
                params,
                lambda fn, elems, axis: associative_scan(fn, elems, axis=axis, schedule=LARGE),
            )

        def reference(params: dict[str, mx.array]) -> mx.array:
            return loss(params, tree_scan)

        first, grads = mx.value_and_grad(compiled)(params)
        _, expected = mx.value_and_grad(reference)(params)
        for key in params:
            self.assertTrue(close(grads[key], expected[key], 1e-3), key)
        step = mx.value_and_grad(compiled)
        value = first
        for _ in range(30):
            value, grads = step(params)
            params = {key: params[key] - 0.5 * grads[key] for key in params}
            mx.eval(params)
        self.assertLess(float(value), 0.5 * float(first))


if __name__ == "__main__":
    unittest.main()
