"""Associative scan on the tiled kernels: lowering without a device, execution on sm_90.

The execution oracle is the generic Blelloch tree in ``experiments/associative_scan``
differentiated by MLX, the way JAX checks its native cumsum gradient against
``associative_scan``. Lengths straddle every level of the hierarchy: one thread's
chunk, one warp, one block, one tile, and several levels of tile recursion.
"""

import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from associative_scan import (
    ScanContractError,
    ScanOp,
    affine_combine,
    associative_scan,
    operation,
)
from graph import capture
from scan_lowering import ScanSchedule, lower_apply, lower_tile_scan

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "associative_scan"))
from scan import associative_scan as tree_scan  # noqa: E402

SMALL = ScanSchedule(threads=32, elements_per_thread=1)
MEDIUM = ScanSchedule(threads=64, elements_per_thread=2)
LARGE = ScanSchedule(threads=128, elements_per_thread=4)
LENGTHS = (1, 2, 3, 5, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1025, 4097)

ON_DEVICE = mx.cuda.is_available() and mx.device_info(mx.gpu)["architecture"] == "sm_90"
device = unittest.skipUnless(ON_DEVICE, "requires MLX CUDA on sm_90")


def affine(left, right):
    """The affine recurrence h_t = a_t h_{t-1} + b_t; non-commutative."""
    a_left, b_left = left
    a_right, b_right = right
    return (a_right * a_left, a_right * b_left + b_right)


def random(shape, seed, scale=1.0):
    return mx.array(
        np.random.default_rng(seed).standard_normal(shape).astype(np.float32) * scale
    )


def close(a, b, tolerance=1e-4):
    return np.allclose(np.array(a), np.array(b), rtol=tolerance, atol=tolerance)


class LoweringTest(unittest.TestCase):
    def test_combine_graph_is_scalar_and_elementwise(self):
        op = operation(affine, ("0", "1"), 1, SMALL)
        self.assertEqual(len(op.graph.inputs), 4)
        self.assertEqual(len(op.graph.outputs), 2)
        self.assertTrue(all(value.shape == () for value in op.graph.inputs))

    def test_tile_kernel_addresses_through_axis_layouts(self):
        op = operation(mx.add, ("",), 0, LARGE)
        lowered = lower_tile_scan(op.graph, (((513, 4), (1, 513)),), 0, LARGE)
        self.assertIn('"((1,4),513):((0,513),1)"', lowered.mlir)
        self.assertIn('"((1,4),513):((0,1),4)"', lowered.mlir)
        self.assertIn('"(4,2):(2,1)"', lowered.mlir)
        self.assertEqual(lowered.grid, (2 * 4 * 128, 1, 1))
        self.assertEqual(lowered.output_shapes, ((513, 4), (4, 2)))
        self.assertEqual(lowered.mlir.count("nvvm.shfl.sync up"), 6)
        self.assertEqual(lowered.shared_memory_bytes, 16)

    def test_single_warp_needs_no_shared_memory(self):
        op = operation(mx.add, ("",), 0, SMALL)
        lowered = lower_tile_scan(op.graph, (((7,), (1,)),), 0, SMALL)
        self.assertNotIn("smem", lowered.mlir)
        self.assertEqual(lowered.shared_memory_bytes, 0)

    def test_apply_kernel_folds_the_previous_tile(self):
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

    def test_contract(self):
        with self.assertRaises(ScanContractError):
            associative_scan(mx.add, [])
        with self.assertRaises(ScanContractError):
            associative_scan(mx.add, mx.zeros((3, 4)), axis=2)
        op = operation(mx.add, ("",), 0, SMALL)
        with self.assertRaises(ScanContractError):
            op.check((mx.zeros((3,), dtype=mx.float16),))


@device
class ForwardTest(unittest.TestCase):
    def test_cumsum_every_length_and_schedule(self):
        for schedule in (SMALL, MEDIUM, LARGE):
            for length in LENGTHS:
                x = random((3, length), length)
                result = associative_scan(mx.add, x, axis=1, schedule=schedule)
                self.assertTrue(
                    close(result, mx.cumsum(x, axis=1), 1e-3), (schedule, length)
                )

    def test_reverse_matches_reverse_cumsum(self):
        for length in (1, 2, 33, 513, 4097):
            x = random((2, length), length)
            result = associative_scan(mx.add, x, axis=1, reverse=True, schedule=MEDIUM)
            self.assertTrue(
                close(result, mx.cumsum(x, axis=1, reverse=True), 1e-3), length
            )

    def test_any_axis_of_a_rank_three_array(self):
        x = random((6, 70, 5), 7)
        for axis in (0, 1, 2, -1):
            result = associative_scan(mx.add, x, axis=axis, schedule=SMALL)
            self.assertTrue(close(result, mx.cumsum(x, axis=axis), 1e-3), axis)

    def test_affine_matches_the_tree(self):
        for length in (1, 2, 3, 64, 65, 1000, 4097):
            a, b = random((4, length), length, 0.9), random((4, length), length + 1)
            result = associative_scan(affine, (a, b), axis=1, schedule=MEDIUM)
            expected = tree_scan(affine, (a, b), axis=1)
            for got, want in zip(result, expected):
                self.assertTrue(close(got, want, 1e-3), length)

    def test_pytree_leaves(self):
        x = random((5, 100), 3)

        def combine(left, right):
            return {
                "sum": left["sum"] + right["sum"],
                "max": mx.maximum(left["max"], right["max"]),
            }

        with self.assertRaises(Exception):
            associative_scan(combine, {"sum": x, "max": x}, axis=1, schedule=SMALL)

    def test_strided_views_are_consumed_in_place(self):
        base = random((300, 7), 5)
        transposed = base.T
        sliced = base[3:, 2:6]
        for x, axis in ((transposed, 1), (sliced, 0), (base[::-1], 0)):
            result = associative_scan(mx.add, x, axis=axis, schedule=MEDIUM)
            self.assertTrue(close(result, mx.cumsum(x, axis=axis), 1e-3))

    def test_empty(self):
        x = mx.zeros((0, 4))
        result = associative_scan(mx.add, x, axis=1)
        self.assertEqual(result.shape, (0, 4))


@device
class DerivativeTest(unittest.TestCase):
    def check_vjp(self, fn, elems, axis, schedule, tolerance=1e-3):
        def compiled(*leaves):
            return list(associative_scan(fn, leaves, axis=axis, schedule=schedule))

        def tree(*leaves):
            return list(tree_scan(fn, leaves, axis=axis))

        cotangents = [
            random(leaf.shape, 100 + index) for index, leaf in enumerate(elems)
        ]
        got = mx.vjp(compiled, list(elems), cotangents)[1]
        want = mx.vjp(tree, list(elems), cotangents)[1]
        for actual, expected in zip(got, want):
            self.assertTrue(
                close(actual, expected, tolerance), (fn.__name__, elems[0].shape)
            )

    def check_jvp(self, fn, elems, axis, schedule, tolerance=1e-3):
        def compiled(*leaves):
            return list(associative_scan(fn, leaves, axis=axis, schedule=schedule))

        def tree(*leaves):
            return list(tree_scan(fn, leaves, axis=axis))

        tangents = [random(leaf.shape, 200 + index) for index, leaf in enumerate(elems)]
        got = mx.jvp(compiled, list(elems), tangents)[1]
        want = mx.jvp(tree, list(elems), tangents)[1]
        for actual, expected in zip(got, want):
            self.assertTrue(
                close(actual, expected, tolerance), (fn.__name__, elems[0].shape)
            )

    def test_cumsum_gradient_is_reverse_cumsum(self):
        for length in (1, 2, 33, 513, 4097):
            x = random((3, length), length)
            grad = mx.grad(
                lambda v: associative_scan(mx.add, v, axis=1, schedule=MEDIUM).sum()
            )(x)
            self.assertTrue(
                close(grad, mx.cumsum(mx.ones_like(x), axis=1, reverse=True), 1e-3),
                length,
            )

    def test_affine_vjp_and_jvp_match_the_tree(self):
        for length in (1, 2, 3, 64, 65, 300, 4097):
            a, b = random((3, length), length, 0.9), random((3, length), length + 1)
            self.check_vjp(affine, (a, b), 1, MEDIUM)
            self.check_jvp(affine, (a, b), 1, MEDIUM)

    def test_complex_affine_couples_every_leaf(self):
        """``z_t = w_t z_{t-1} + v_t`` over complex ``w`` and ``v`` as four real
        leaves: associative, non-commutative, and every Jacobian block is dense."""

        def complex_affine(left, right):
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
        for got, want in zip(result, tree_scan(complex_affine, x, axis=1)):
            self.assertTrue(close(got, want, 1e-3))
        self.check_vjp(complex_affine, x, 1, SMALL)
        self.check_jvp(complex_affine, x, 1, SMALL)

    def test_reverse_derivatives(self):
        a, b = random((2, 300), 1, 0.9), random((2, 300), 2)

        def compiled(a, b):
            return list(
                associative_scan(affine, (a, b), axis=1, reverse=True, schedule=MEDIUM)
            )

        def tree(a, b):
            return list(tree_scan(affine, (a, b), axis=1, reverse=True))

        cotangents = [random((2, 300), 3), random((2, 300), 4)]
        got = mx.vjp(compiled, [a, b], cotangents)[1]
        want = mx.vjp(tree, [a, b], cotangents)[1]
        for actual, expected in zip(got, want):
            self.assertTrue(close(actual, expected, 1e-3))

    def test_derivatives_through_strided_views(self):
        base = random((300, 3), 9, 0.9)
        other = random((300, 3), 10)

        def compiled(base, other):
            return list(
                associative_scan(affine, (base.T, other.T), axis=1, schedule=MEDIUM)
            )

        def tree(base, other):
            return list(tree_scan(affine, (base.T, other.T), axis=1))

        cotangents = [random((3, 300), 11), random((3, 300), 12)]
        got = mx.vjp(compiled, [base, other], cotangents)[1]
        want = mx.vjp(tree, [base, other], cotangents)[1]
        for actual, expected in zip(got, want):
            self.assertTrue(close(actual, expected, 1e-3))

    def test_training_a_linear_recurrence(self):
        """A diagonal linear SSM trained with value_and_grad; the first step's
        gradient equals the tree's and the loss falls over the run."""
        batch, length, width = 4, 700, 8
        inputs = random((batch, length, width), 20)
        teacher = {"decay": random((width,), 21, 0.5), "gain": random((width,), 22)}

        def model(params, inputs, scan):
            decay = mx.sigmoid(params["decay"]) * mx.ones_like(inputs)
            drive = inputs * params["gain"]
            _, hidden = scan(affine, (decay, drive), axis=1)
            return hidden

        target = model(teacher, inputs, tree_scan)
        params = {"decay": mx.zeros((width,)), "gain": mx.ones((width,))}

        def loss(params, scan):
            return mx.mean((model(params, inputs, scan) - target) ** 2)

        compiled = lambda params: loss(
            params,
            lambda fn, elems, axis: associative_scan(
                fn, elems, axis=axis, schedule=LARGE
            ),
        )
        reference = lambda params: loss(params, tree_scan)
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
