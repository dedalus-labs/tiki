"""Associative scan on the tiled kernels: lowering without a device, execution on sm_90.

The execution oracle is the generic Blelloch tree in ``experiments/associative_scan``
differentiated by MLX, the way JAX checks its native cumsum gradient against
``associative_scan``. Lengths straddle every level of the hierarchy: one thread's
chunk, one warp, one block, one tile, and several levels of tile recursion.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.tiki.testing import check_grads

import tiki as tk
from associative_scan import (
    ScanContractError,
    ScanOp,
    TreeDef,
    affine_combine,
    associative_scan,
    operation,
)
from graph import capture
from lowering import UnsupportedScheduleError
from scan_lowering import ScanSchedule, lower_apply, lower_tile_scan
from tiki import Compiled

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
        op = operation(affine, TreeDef(tuple, (TreeDef(mx.array),) * 2), 2, 1, SMALL)
        self.assertEqual(len(op.graph.inputs), 4)
        self.assertEqual(len(op.graph.outputs), 2)
        self.assertTrue(all(value.shape == () for value in op.graph.inputs))

    def test_tile_kernel_addresses_through_axis_layouts(self):
        op = operation(mx.add, TreeDef(mx.array), 1, 0, LARGE)
        lowered = lower_tile_scan(op.graph, (((513, 4), (1, 513)),), 0, LARGE)
        self.assertIn('"((1,4),513):((0,513),1)"', lowered.mlir)
        self.assertIn('"((1,4),513):((0,1),4)"', lowered.mlir)
        self.assertIn('"(4,2):(2,1)"', lowered.mlir)
        self.assertEqual(lowered.grid, (2 * 4 * 128, 1, 1))
        self.assertEqual(lowered.output_shapes, ((513, 4), (4, 2)))
        self.assertEqual(lowered.mlir.count("nvvm.shfl.sync up"), 6)
        self.assertEqual(lowered.shared_memory_bytes, 16)

    def test_single_warp_needs_no_shared_memory(self):
        op = operation(mx.add, TreeDef(mx.array), 1, 0, SMALL)
        lowered = lower_tile_scan(op.graph, (((7,), (1,)),), 0, SMALL)
        self.assertNotIn("smem", lowered.mlir)
        self.assertEqual(lowered.shared_memory_bytes, 0)

    def test_apply_kernel_folds_the_previous_tile(self):
        op = operation(affine, TreeDef(tuple, (TreeDef(mx.array),) * 2), 2, 1, MEDIUM)
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

    def test_launch_grid_fits_signed_runtime_dimensions(self):
        # One thread block per singleton row reaches 2**31 at 16,777,216 rows.
        graph = capture(mx.add, (((), ()),) * 2)
        for schedule in (SMALL, LARGE):
            rows = 2**31 // schedule.threads
            for lower in (
                lambda shape: lower_tile_scan(graph, ((shape, (1, 1)),), 1, schedule),
                lambda shape: lower_apply(graph, shape, 1, 1, schedule),
            ):
                with self.assertRaisesRegex(UnsupportedScheduleError, "signed 32-bit"):
                    lower((rows, 1))
                self.assertEqual(
                    lower((rows - 1, 1)).grid,
                    ((rows - 1) * schedule.threads, 1, 1),
                )

    def test_contract(self):
        with self.assertRaises(ScanContractError):
            associative_scan(mx.add, [])
        with self.assertRaises(ScanContractError):
            associative_scan(mx.add, mx.zeros((3, 4)), axis=2)
        op = operation(mx.add, TreeDef(mx.array), 1, 0, SMALL)
        with self.assertRaises(ScanContractError):
            op.check((mx.zeros((3,), dtype=mx.float16),))


class StructureTest(unittest.TestCase):
    def test_tuple_structure_is_preserved(self):
        # Tuple leaves must stay tuples in fn and its result, even when nested.
        x = mx.array([1.0, 2.0, 3.0])

        def combine(left, right):
            self.assertIsInstance(left, tuple)
            self.assertIsInstance(right, tuple)
            self.assertIsInstance(left[1], list)
            return (left[0] + right[0], [left[1][0] + right[1][0]])

        with patch.object(ScanOp, "__call__", lambda op, *leaves: leaves):
            result = associative_scan(combine, (x, [x]))
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[1], list)

    def test_dictionary_order_does_not_change_structure(self):
        # Keys identify leaves, even if fn inserts the same keys in reverse order.
        x = mx.array([1.0, 2.0, 3.0])

        def combine(left, right):
            return {"b": left["b"] + right["b"], "a": left["a"] + right["a"]}

        with patch.object(
            ScanOp, "__call__", lambda op, *leaves: op.combine(*leaves, *leaves)
        ):
            result = associative_scan(combine, {"a": x, "b": 2 * x})
        self.assertTrue(close(result["a"], 2 * x))
        self.assertTrue(close(result["b"], 4 * x))

    def test_changed_container_type_is_rejected(self):
        # Matching leaf paths do not permit fn to replace a tuple with a list.
        x = mx.array([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ScanContractError, "structure of elems"):
            associative_scan(lambda left, right: [left[0] + right[0]], (x,))

    def test_dictionary_keys_and_empty_nodes_are_preserved(self):
        # Literal dotted keys and empty nodes are part of the input structure.
        x = mx.array([1.0, 2.0, 3.0])
        elems = {"a.b": (x, []), "0": {}}
        with patch.object(ScanOp, "__call__", lambda op, *leaves: leaves):
            result = associative_scan(lambda left, right: left, elems)
        self.assertEqual(set(result), set(elems))
        self.assertIsInstance(result["a.b"], tuple)
        self.assertEqual(result["a.b"][1], [])
        self.assertEqual(result["0"], {})


class CallbackTest(unittest.TestCase):
    def test_inactive_scan_tangents_are_zero(self):
        # Inactive leaves contribute zero, with either affine input held fixed.
        def forward(op, *leaves):
            return tuple(
                tree_scan(
                    lambda left, right: op.combine(*left, *right), leaves, axis=op.axis
                )
            )

        a, b = random((1, 3), 1, 0.9), random((1, 3), 2)
        with patch.object(ScanOp, "forward", forward), patch.object(
            Compiled, "launch", lambda kernel, *args: kernel.function(*args)
        ):
            op = ScanOp(lambda al, bl, ar, br: affine((al, bl), (ar, br)), 2, 1, SMALL)
            for active in range(2):

                def reference(value):
                    leaves = (value, b) if active == 0 else (a, value)
                    a_values, b_values = leaves
                    hidden = [b_values[:, 0]]
                    for index in range(1, 3):
                        hidden.append(
                            a_values[:, index] * hidden[-1] + b_values[:, index]
                        )
                    return mx.stack(hidden, axis=1)

                primal = (a, b)[active]
                tangent = mx.ones_like(primal)
                tangents = (tangent, None) if active == 0 else (None, tangent)
                got = op._jvp((a, b), tangents)[1]
                want = mx.jvp(reference, [primal], [tangent])[1][0]
                self.assertTrue(close(got, want))


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

    def test_check_grads_against_the_tree(self):
        """The generic checker on the real op: forward, reverse, vmap, and every
        layout, with the tree as the oracle, at lengths inside and beyond a tile."""
        for length in (5, 300):
            a, b = random((3, length), length, 0.9), random((3, length), length + 1)
            check_grads(
                lambda a, b: associative_scan(affine, (a, b), axis=1, schedule=MEDIUM),
                (a, b),
                reference=lambda a, b: tree_scan(affine, (a, b), axis=1),
            )

    def test_check_grads_on_a_compiled_elementwise_region(self):
        function = lambda x, y: x * y + 2.0 - mx.rsqrt(y * y + 1.0)
        compiled = tk.compile()(function)
        check_grads(
            compiled,
            (random((7, 9), 30), random((7, 9), 31)),
            reference=function,
            order=2,
        )

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
