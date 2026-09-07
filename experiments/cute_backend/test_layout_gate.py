"""The first stride-system gate: views consumed in place with correct derivatives.

Each test states its invariant and witness. Execution tests need MLX CUDA on
sm_90 and an MLX build that exposes array strides.
"""

import unittest

import mlx.core as mx
import numpy as np

import tiki as tk
from graph import ArrayFunction, Value, dense_strides

HAS_STRIDES = hasattr(mx.array, "strides")
CAN_EXECUTE = (
    HAS_STRIDES
    and mx.cuda.is_available()
    and mx.device_info(mx.gpu)["architecture"] == "sm_90"
)


def affine(x: mx.array, y: mx.array) -> mx.array:
    return x * y + 2.0 - y


def values(array: mx.array) -> np.ndarray:
    mx.eval(array)
    mx.synchronize()
    return np.asarray(array)


def peak_growth(function: ArrayFunction, *args: mx.array) -> tuple[mx.array, int]:
    """Run ``function`` and report how far peak memory rose above the start."""
    warm = function(*args)
    mx.eval(warm)
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    result = function(*args)
    mx.eval(result)
    mx.synchronize()
    return result, mx.get_peak_memory() - before


class TestLoweringWithoutDevice(unittest.TestCase):
    # Invariant: a dense value lowers to the flat memref it always did, so
    # existing schedules and audited MLIR are byte-for-byte unchanged; a
    # strided value lowers to its own (shape):(strides) memref.
    # Witness: a dense and a transposed 64x513 value.
    def test_memref_forms(self):
        from lowering import memref

        dense = Value("a", (64, 513))
        self.assertEqual(dense.strides, dense_strides((64, 513)))
        self.assertEqual(
            memref(dense), '!cute.memref<f32, gmem, align<4>, "(32832):(1)">'
        )
        transposed = Value("t", (64, 513), (1, 64))
        self.assertFalse(transposed.is_dense)
        self.assertEqual(
            memref(transposed), '!cute.memref<f32, gmem, align<4>, "(64,513):(1,64)">'
        )

    # Invariant: dense profiles produce the same MLIR as before this change.
    # Witness: the elementwise demo graph at (513,) contains no logical
    # coordinate and only flat memrefs.
    def test_dense_mlir_is_unchanged(self):
        lowered = tk.specialize(affine, tk.Schedule(), (((513,), (1,)), ((513,), (1,))))
        self.assertNotIn("%logical", lowered.mlir)
        self.assertIn('"(513):(1)"', lowered.mlir)
        self.assertNotIn(":(1,", lowered.mlir)

    # Invariant: a strided profile adds the logical coordinate and addresses
    # the strided input through its layout while the dense input keeps the
    # flat form.
    # Witness: x transposed, y dense, both (64, 513).
    def test_strided_mlir(self):
        lowered = tk.specialize(
            affine, tk.Schedule(), (((64, 513), (1, 64)), ((64, 513), (513, 1)))
        )
        self.assertIn("%logical = cute.make_coord(%i0, %i1)", lowered.mlir)
        self.assertIn('"(64,513):(1,64)"', lowered.mlir)
        self.assertIn("%input1 = cute.memref.load(%arg1, %coord)", lowered.mlir)

    # Invariant: cooperative schedules pack views explicitly and specialize on
    # the dense profile, so a transposed input lowers to the same kernel as a
    # dense one; only the elementwise schedule addresses views in place.
    # Witness: a row schedule over a transposed and over a dense input.
    def test_cooperative_schedules_pack_views(self):
        def rms(x: mx.array, w: mx.array) -> mx.array:
            return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w

        schedule = tk.RowSchedule(threads_per_row=32, rows_per_block=4)
        strided = tk.specialize(rms, schedule, (((8, 64), (1, 8)), ((64,), (1,))))
        dense = tk.specialize(rms, schedule, (((8, 64), (64, 1)), ((64,), (1,))))
        self.assertEqual(strided.mlir, dense.mlir)
        self.assertTrue(tk.packs_views(schedule))
        self.assertFalse(tk.packs_views(tk.Schedule()))


@unittest.skipUnless(CAN_EXECUTE, "needs MLX CUDA on sm_90 with array strides")
class TestGate(unittest.TestCase):
    def setUp(self) -> None:
        self.compiled = tk.compile()(affine)
        rng = np.random.default_rng(7)
        self.a = mx.array(rng.normal(size=(513, 64)).astype(np.float32))
        self.y = mx.array(rng.normal(size=(64, 513)).astype(np.float32))
        mx.eval(self.a, self.y)

    def test_derivatives_use_the_frozen_forward_graph(self) -> None:
        coefficient = [2.0]
        compiled = tk.compile()(lambda x: x * coefficient[0])
        x = mx.array([1.0, 2.0, 3.0])
        mx.eval(compiled(x))
        coefficient[0] = 3.0
        outputs, gradients = mx.vjp(compiled, (x,), (mx.ones_like(x),))
        np.testing.assert_array_equal(values(outputs[0]), values(2 * x))
        np.testing.assert_array_equal(
            values(gradients[0]), values(mx.full(x.shape, 2.0))
        )
        tangent = mx.jvp(compiled, (x,), (mx.ones_like(x),))[1][0]
        np.testing.assert_array_equal(values(tangent), values(mx.full(x.shape, 2.0)))

    def test_derivative_cache_tracks_input_arity(self) -> None:
        def function(x: mx.array, y: mx.array | None = None) -> mx.array:
            return x * x if y is None else x * y

        compiled = tk.compile()(function)
        x, y = mx.array([2.0, 3.0]), mx.array([5.0, 7.0])
        mx.eval(mx.vjp(compiled, (x,), (mx.ones_like(x),))[1])
        got = mx.vjp(compiled, (x, y), (mx.ones_like(x),))[1]
        for actual, expected in zip(got, (y, x)):
            np.testing.assert_array_equal(values(actual), values(expected))

    def test_tape_owns_its_specialization_after_cache_eviction(self) -> None:
        coefficient = [2.0]
        compiled = tk.compile()(lambda x: x * coefficient[0])

        def outer(x: mx.array) -> mx.array:
            output = compiled(x)
            tk.specialize.cache_clear()
            tk.differentiable.cache_clear()
            coefficient[0] = 3.0
            return output

        x = mx.array([1.0, 2.0])
        outputs, gradients = mx.vjp(outer, (x,), (mx.ones_like(x),))
        np.testing.assert_array_equal(values(outputs[0]), values(2 * x))
        np.testing.assert_array_equal(
            values(gradients[0]), values(mx.full(x.shape, 2.0))
        )

    def test_registered_kernels_remain_differentiable(self) -> None:
        square = tk.compile()(lambda x: x * x)
        gradient = mx.grad(lambda x: mx.sum(square(x)))
        x = mx.array([1.0, 2.0, 3.0])
        second = mx.grad(lambda x: mx.sum(gradient(x)))(x)
        np.testing.assert_array_equal(values(second), values(mx.full(x.shape, 2.0)))

    # Invariant: a transposed input is consumed in place: the result matches
    # eager MLX and peak memory rises by no more than the output.
    # Witness: x = a.T of shape (64, 513) with a dense y.
    def test_transposed_input_without_packing(self):
        x = self.a.T
        self.assertEqual(x.strides, (1, 64))
        dense, dense_growth = peak_growth(self.compiled, self.y, self.y)
        result, growth = peak_growth(self.compiled, x, self.y)
        np.testing.assert_allclose(
            values(result), values(affine(x, self.y)), rtol=1e-6, atol=1e-6
        )
        self.assertLessEqual(
            growth,
            dense_growth,
            f"view grew {growth} bytes, dense control grew {dense_growth}",
        )

    # Invariant: a sliced input with a nonzero offset is consumed in place.
    # Witness: x = big[3:, 5:300] of shape (61, 295) against a dense y.
    def test_sliced_input_without_packing(self):
        big = mx.array(
            np.random.default_rng(3).normal(size=(64, 305)).astype(np.float32)
        )
        x = big[3:, 5:300]
        y = mx.array(np.random.default_rng(4).normal(size=x.shape).astype(np.float32))
        mx.eval(big, x, y)
        self.assertNotEqual(x.offset, 0)
        dense, dense_growth = peak_growth(self.compiled, y, y)
        result, growth = peak_growth(self.compiled, x, y)
        np.testing.assert_allclose(
            values(result), values(affine(x, y)), rtol=1e-6, atol=1e-6
        )
        self.assertLessEqual(
            growth,
            dense_growth,
            f"view grew {growth} bytes, dense control grew {dense_growth}",
        )

    def test_allocation_gate_detects_an_input_copy(self) -> None:
        def packed(x: mx.array, y: mx.array) -> mx.array:
            return self.compiled(mx.contiguous(x, allow_col_major=False), y)

        dense, dense_growth = peak_growth(self.compiled, self.y, self.y)
        result, growth = peak_growth(packed, self.a.T, self.y)
        self.assertGreater(growth, dense_growth)

    # Invariant: equal shapes with different layouts are separate kernels.
    # Witness: the dense y and the transposed view both of shape (64, 513).
    def test_cache_separates_layouts(self):
        tk.specialize.cache_clear()
        mx.eval(self.compiled(self.y, self.y))
        mx.eval(self.compiled(self.a.T, self.y))
        self.assertEqual(tk.specialize.cache_info().currsize, 2)

    # Invariant: the registered VJP equals MLX's eager VJP for arbitrary
    # cotangents on transposed and sliced inputs.
    # Witness: random cotangents, both inputs.
    def test_vjp_matches_eager(self):
        x = self.a.T
        cotangent = mx.array(
            np.random.default_rng(9).normal(size=x.shape).astype(np.float32)
        )
        mx.eval(cotangent)
        compiled_grads = mx.vjp(self.compiled, (x, self.y), (cotangent,))[1]
        eager_grads = mx.vjp(affine, (x, self.y), (cotangent,))[1]
        for got, want in zip(compiled_grads, eager_grads):
            np.testing.assert_allclose(values(got), values(want), rtol=1e-6, atol=1e-6)

    # Invariant: derivatives of a single-input compiled region follow MLX's
    # bare-array callback convention.
    # Witness: square of a transposed view, VJP and JVP against eager.
    def test_single_input_derivatives(self):
        square = tk.compile()(lambda x: x * x)
        x = self.a.T
        cotangent = mx.ones_like(x)
        got = mx.vjp(square, (x,), (cotangent,))[1][0]
        want = mx.vjp(lambda x: x * x, (x,), (cotangent,))[1][0]
        np.testing.assert_allclose(values(got), values(want), rtol=1e-6, atol=1e-6)
        got = mx.jvp(square, (x,), (cotangent,))[1][0]
        want = mx.jvp(lambda x: x * x, (x,), (cotangent,))[1][0]
        np.testing.assert_allclose(values(got), values(want), rtol=1e-6, atol=1e-6)

    # Invariant: the registered JVP equals MLX's eager JVP.
    # Witness: unit tangents on a transposed input.
    def test_jvp_matches_eager(self):
        x = self.a.T
        tangents = (mx.ones_like(x), mx.ones_like(self.y))
        got = mx.jvp(self.compiled, (x, self.y), tangents)[1][0]
        want = mx.jvp(affine, (x, self.y), tangents)[1][0]
        np.testing.assert_allclose(values(got), values(want), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
