"""The first stride-system gate: views consumed in place with correct derivatives.

Each test states its invariant and witness. Execution tests need Tiki CUDA on
sm_90 and an Tiki build that exposes array strides.
"""

import unittest

import tiki as tk
import numpy as np
from numpy.typing import NDArray

import compiler
from tiki_compiler.arrays import single
from tiki_compiler.execution import packs_views
from tiki_compiler.graph import ArrayFunction, Profile, Symbol, Value, dense_strides
from tiki_compiler.lowered import Lowered
from tiki_compiler.scalar import memref
from tiki_compiler.specialize import specialize

HAS_STRIDES = hasattr(tk.array, "strides")
CAN_EXECUTE = (
    HAS_STRIDES and tk.cuda.is_available() and tk.device_info(tk.gpu)["architecture"] == "sm_90"
)


def affine(x: tk.array, y: tk.array) -> tk.array:
    return x * y + 2.0 - y


def values(array: tk.array) -> NDArray[np.float32]:
    tk.eval(array)
    tk.synchronize()
    return np.asarray(array)


def peak_growth(function: ArrayFunction, *args: tk.array) -> tuple[tk.array, int]:
    """Run ``function`` and report how far peak memory rose above the start."""
    tk.synchronize()
    tk.clear_cache()
    before = tk.get_active_memory()
    tk.reset_peak_memory()
    result = single(function(*args))
    tk.eval(result)
    tk.synchronize()
    return result, tk.get_peak_memory() - before


class TestLoweringWithoutDevice(unittest.TestCase):
    # Invariant: a dense value lowers to the flat memref it always did, so
    # existing schedules and audited MLIR are byte-for-byte unchanged; a
    # strided value lowers to its own (shape):(strides) memref.
    # Witness: a dense and a transposed 64x513 value.
    def test_memref_forms(self) -> None:

        dense = Value.dense(Symbol("a"), (64, 513))
        self.assertEqual(dense.strides, dense_strides((64, 513)))
        self.assertEqual(memref(dense), '!cute.memref<f32, gmem, align<4>, "(32832):(1)">')
        transposed = Value(name=Symbol("t"), shape=(64, 513), strides=(1, 64))
        self.assertFalse(transposed.is_dense)
        self.assertEqual(memref(transposed), '!cute.memref<f32, gmem, align<4>, "(64,513):(1,64)">')

    # Invariant: dense profiles produce the same MLIR as before this change.
    # Witness: the elementwise demo graph at (513,) contains no logical
    # coordinate and only flat memrefs.
    def test_dense_mlir_is_unchanged(self) -> None:
        lowered = specialize(
            affine,
            compiler.Schedule(),
            (Profile(shape=(513,), strides=(1,)), Profile(shape=(513,), strides=(1,))),
            tk.default_stream(tk.default_device()),
        )
        assert isinstance(lowered, Lowered)
        self.assertNotIn("%logical", lowered.mlir)
        self.assertIn('"(513):(1)"', lowered.mlir)
        self.assertNotIn(":(1,", lowered.mlir)

    # Invariant: a strided profile adds the logical coordinate and addresses
    # the strided input through its layout while the dense input keeps the
    # flat form.
    # Witness: x transposed, y dense, both (64, 513).
    def test_strided_mlir(self) -> None:
        lowered = specialize(
            affine,
            compiler.Schedule(),
            (Profile(shape=(64, 513), strides=(1, 64)), Profile(shape=(64, 513), strides=(513, 1))),
            tk.default_stream(tk.default_device()),
        )
        assert isinstance(lowered, Lowered)
        self.assertIn("%logical = cute.make_coord(%i0, %i1)", lowered.mlir)
        self.assertIn('"(64,513):(1,64)"', lowered.mlir)
        self.assertIn("%input1 = cute.memref.load(%arg1, %coord)", lowered.mlir)

    # Invariant: cooperative schedules pack views explicitly and specialize on
    # the dense profile, so a transposed input lowers to the same kernel as a
    # dense one; only the elementwise schedule addresses views in place.
    # Witness: a row schedule over a transposed and over a dense input.
    def test_cooperative_schedules_pack_views(self) -> None:
        def rms(x: tk.array, w: tk.array) -> tk.array:
            return x * tk.rsqrt(tk.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w

        schedule = compiler.RowSchedule(threads_per_row=32, rows_per_block=4)
        stream = tk.default_stream(tk.default_device())
        strided = specialize(
            rms,
            schedule,
            (Profile(shape=(8, 64), strides=(1, 8)), Profile(shape=(64,), strides=(1,))),
            stream,
        )
        assert isinstance(strided, Lowered)
        dense = specialize(
            rms,
            schedule,
            (Profile(shape=(8, 64), strides=(64, 1)), Profile(shape=(64,), strides=(1,))),
            stream,
        )
        assert isinstance(dense, Lowered)
        self.assertEqual(strided.mlir, dense.mlir)
        self.assertTrue(packs_views(schedule))
        self.assertFalse(packs_views(compiler.Schedule()))


@unittest.skipUnless(CAN_EXECUTE, "needs Tiki CUDA on sm_90 with array strides")
class TestGate(unittest.TestCase):
    def setUp(self) -> None:
        self.compiled = compiler.compile()(affine)
        rng = np.random.default_rng(7)
        self.a = tk.array(rng.normal(size=(513, 64)).astype(np.float32))
        self.y = tk.array(rng.normal(size=(64, 513)).astype(np.float32))
        tk.eval(self.a, self.y)

    # Invariant: a transposed input is consumed in place: the result matches
    # eager Tiki and peak memory rises by no more than the output.
    # Witness: x = a.T of shape (64, 513) with a dense y.
    def test_transposed_input_without_packing(self) -> None:
        x = self.a.T
        self.assertEqual(x.strides, (1, 64))
        result, growth = peak_growth(self.compiled, x, self.y)
        np.testing.assert_allclose(values(result), values(affine(x, self.y)), rtol=1e-6, atol=1e-6)
        self.assertLessEqual(
            growth, 2 * result.nbytes, f"packing suspected: peak grew {growth} bytes"
        )

    # Invariant: a sliced input with a nonzero offset is consumed in place.
    # Witness: x = big[3:, 5:300] of shape (61, 295) against a dense y.
    def test_sliced_input_without_packing(self) -> None:
        big = tk.array(np.random.default_rng(3).normal(size=(64, 305)).astype(np.float32))
        x = big[3:, 5:300]
        y = tk.array(np.random.default_rng(4).normal(size=x.shape).astype(np.float32))
        tk.eval(big, x, y)
        self.assertNotEqual(x.offset, 0)
        result, growth = peak_growth(self.compiled, x, y)
        np.testing.assert_allclose(values(result), values(affine(x, y)), rtol=1e-6, atol=1e-6)
        self.assertLessEqual(
            growth, 2 * result.nbytes, f"packing suspected: peak grew {growth} bytes"
        )

    # Invariant: equal shapes with different layouts are separate kernels.
    # Witness: the dense y and the transposed view both of shape (64, 513).
    def test_cache_separates_layouts(self) -> None:
        specialize.cache_clear()
        tk.eval(self.compiled(self.y, self.y))
        tk.eval(self.compiled(self.a.T, self.y))
        self.assertEqual(specialize.cache_info().currsize, 2)

    # Invariant: the registered VJP equals Tiki's eager VJP for arbitrary
    # cotangents on transposed and sliced inputs.
    # Witness: random cotangents, both inputs.
    def test_vjp_matches_eager(self) -> None:
        x = self.a.T
        cotangent = tk.array(np.random.default_rng(9).normal(size=x.shape).astype(np.float32))
        tk.eval(cotangent)
        compiled_grads = tk.vjp(self.compiled, [x, self.y], [cotangent])[1]
        eager_grads = tk.vjp(affine, [x, self.y], [cotangent])[1]
        for got, want in zip(compiled_grads, eager_grads, strict=True):
            np.testing.assert_allclose(values(got), values(want), rtol=1e-6, atol=1e-6)

    # Invariant: derivatives of a single-input compiled region follow Tiki's
    # bare-array callback convention.
    # Witness: square of a transposed view, VJP and JVP against eager.
    def test_single_input_derivatives(self) -> None:
        square = compiler.compile()(lambda x: x * x)
        x = self.a.T
        cotangent = tk.ones_like(x)
        got = tk.vjp(square, [x], [cotangent])[1][0]
        want = tk.vjp(lambda x: x * x, [x], [cotangent])[1][0]
        np.testing.assert_allclose(values(got), values(want), rtol=1e-6, atol=1e-6)
        got = tk.jvp(square, [x], [cotangent])[1][0]
        want = tk.jvp(lambda x: x * x, [x], [cotangent])[1][0]
        np.testing.assert_allclose(values(got), values(want), rtol=1e-6, atol=1e-6)

    # Invariant: the registered JVP equals Tiki's eager JVP.
    # Witness: unit tangents on a transposed input.
    def test_jvp_matches_eager(self) -> None:
        x = self.a.T
        tangents = [tk.ones_like(x), tk.ones_like(self.y)]
        got = tk.jvp(self.compiled, [x, self.y], tangents)[1][0]
        want = tk.jvp(affine, [x, self.y], tangents)[1][0]
        np.testing.assert_allclose(values(got), values(want), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
