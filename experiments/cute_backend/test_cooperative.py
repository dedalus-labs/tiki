"""Contracts for cooperative reductions and swizzled shared-memory tiles."""

import unittest

import mlx.core as mx

import tiki as tk


def rms_norm(x, weight):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6) * weight


class ScheduleTests(unittest.TestCase):
    def test_subwarp_rows_keep_reductions_inside_their_thread_groups(self):
        for threads, rows in ((8, 16), (16, 8)):
            compiled = tk.compile(
                schedule=tk.RowSchedule(threads_per_row=threads, rows_per_block=rows)
            )(rms_norm)
            lowered = compiled.lower(mx.zeros((5, 31)), mx.ones((31,)))
            self.assertEqual(lowered.shared_memory_bytes, 0)
            self.assertEqual(
                lowered.mlir.count("nvvm.shfl.sync"), threads.bit_length() - 1
            )

    def test_row_schedule_rejects_transpose_even_for_single_column(self):
        compiled = tk.compile(schedule=tk.RowSchedule())(lambda x: (x * x).T)
        with self.assertRaises(tk.UnsupportedGraphError):
            compiled.lower(mx.ones((1, 1)))

    def test_single_column_rows_survive_reduction_simplification(self):
        compiled = tk.compile(schedule=tk.RowSchedule())(rms_norm)
        lowered = compiled.lower(mx.zeros((5, 1)), mx.ones((1,)))
        self.assertEqual(lowered.shared_memory_bytes, 0)
        self.assertNotIn("nvvm.shfl", lowered.mlir)

    def test_row_schedule_captures_sum_axis_and_rsqrt(self):
        compiled = tk.compile(
            schedule=tk.RowSchedule(threads_per_row=64, rows_per_block=2)
        )(rms_norm)
        lowered = compiled.lower(mx.zeros((3, 129)), mx.ones((129,)))
        self.assertEqual(lowered.grid, (256, 1, 1))
        self.assertEqual(lowered.shared_memory_bytes, 16)
        self.assertIn("ReduceSum", [node.operation for node in lowered.graph.nodes])
        self.assertIn("Rsqrt", [node.operation for node in lowered.graph.nodes])

    def test_unsupported_reduction_axis_and_kind_fail(self):
        for function in (
            lambda x: x + mx.sum(x, axis=0, keepdims=True),
            lambda x: x + mx.max(x, axis=-1, keepdims=True),
        ):
            with self.assertRaises(tk.UnsupportedGraphError):
                tk.compile(schedule=tk.RowSchedule())(function).lower(mx.ones((3, 129)))

    def test_swizzle_is_a_bijection_and_changes_column_banks(self):
        plain = tk.Swizzle(bits=0)
        xor = tk.Swizzle(bits=5)
        for swizzle in (plain, xor, tk.Swizzle(bits=3, base=2)):
            offsets = [
                swizzle.offset(row * 32 + col) for row in range(32) for col in range(32)
            ]
            self.assertEqual(sorted(offsets), list(range(1024)))
        self.assertEqual(len({plain.offset(row * 32) % 32 for row in range(32)}), 1)
        self.assertEqual(len({xor.offset(row * 32) % 32 for row in range(32)}), 32)

    def test_bad_schedules_are_rejected(self):
        for create in (
            lambda: tk.RowSchedule(threads_per_row=48),
            lambda: tk.RowSchedule(threads_per_row=256, rows_per_block=8),
            lambda: tk.RowSchedule(threads_per_row=8, rows_per_block=1),
            lambda: tk.Swizzle(bits=5, base=1),
        ):
            with self.assertRaises(tk.UnsupportedScheduleError):
                create()


@unittest.skipUnless(mx.cuda.is_available(), "requires MLX CUDA")
class CooperativeExecutionTests(unittest.TestCase):
    def test_row_schedule_preserves_offset_and_padded_input_views(self):
        compiled = tk.compile(
            schedule=tk.RowSchedule(threads_per_row=16, rows_per_block=8)
        )(rms_norm)
        weight = mx.arange(34, dtype=mx.float32)[1:]
        inputs = (
            mx.arange(166, dtype=mx.float32)[1:].reshape(5, 33),
            mx.arange(200, dtype=mx.float32).reshape(5, 40)[:, :33],
        )
        for x in inputs:
            self.assertTrue(
                mx.allclose(
                    compiled(x, weight), rms_norm(x, weight), atol=2e-6, rtol=2e-5
                )
            )

    def test_rmsnorm_matches_reference_across_thread_schedules(self):
        for threads, rows in ((8, 16), (16, 8), (32, 4), (64, 2), (128, 1), (256, 1)):
            compiled = tk.compile(
                schedule=tk.RowSchedule(threads_per_row=threads, rows_per_block=rows)
            )(rms_norm)
            for width in (1, 31, 129, 1024, 4096):
                with self.subTest(threads=threads, rows=rows, width=width):
                    x = mx.random.normal((5, width))
                    weight = mx.random.normal((width,))
                    self.assertTrue(
                        mx.allclose(
                            compiled(x, weight),
                            rms_norm(x, weight),
                            atol=2e-6,
                            rtol=2e-5,
                        )
                    )

    def test_zero_rows_and_zero_values_are_well_defined(self):
        compiled = tk.compile(schedule=tk.RowSchedule())(rms_norm)
        for rows in (0, 3):
            x = mx.zeros((rows, 129))
            out = compiled(x, mx.ones((129,)))
            self.assertTrue(mx.array_equal(out, x))

    def test_swizzled_transpose_preserves_values_and_partial_tiles(self):
        for bits, base in ((0, 0), (5, 0), (3, 2)):
            compiled = tk.compile(
                schedule=tk.TransposeSchedule(swizzle=tk.Swizzle(bits=bits, base=base))
            )(lambda x: x.T)
            for shape in ((32, 32), (33, 65), (1, 7), (0, 32)):
                with self.subTest(bits=bits, base=base, shape=shape):
                    x = mx.arange(shape[0] * shape[1], dtype=mx.float32).reshape(shape)
                    self.assertTrue(mx.array_equal(compiled(x), x.T))


if __name__ == "__main__":
    unittest.main()
