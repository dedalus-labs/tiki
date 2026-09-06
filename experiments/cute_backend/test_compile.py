"""Contracts for native MLX graph lowering and CuTe execution."""

import unittest

import mlx.core as mx

import tiki as tk


class CaptureTests(unittest.TestCase):
    def test_native_graph_preserves_arithmetic_and_scalar_broadcast(self):
        function = tk.compile()(lambda x, y: x * y + 2.0 - y)
        lowered = function.lower(mx.zeros((7,)), mx.zeros((7,)))
        self.assertEqual(lowered.graph.shape, (7,))
        self.assertEqual(
            [node.operation for node in lowered.graph.nodes],
            ["Multiply", "Broadcast", "Add", "Subtract"],
        )
        self.assertIn("arith.mulf", lowered.mlir)
        self.assertIn("cute.memref.store", lowered.mlir)

    def test_schedule_controls_thread_mapping(self):
        function = tk.compile(schedule=tk.Schedule(threads=64, elements_per_thread=4))(
            lambda x: x * x
        )
        lowered = function.lower(mx.zeros((257,)))
        self.assertEqual(lowered.grid, (128, 1, 1))
        self.assertIn("array<i32: 64, 1, 1>", lowered.mlir)
        self.assertEqual(lowered.mlir.count("cute.memref.store"), 4)

    def test_unsupported_graphs_fail_explicitly(self):
        for function in (lambda x: mx.sum(x), lambda x: mx.exp(x), lambda x: x.T):
            with self.subTest(function=function), self.assertRaises(
                tk.UnsupportedGraphError
            ):
                tk.compile()(function).lower(mx.zeros((2, 3)))

    def test_dtype_and_non_scalar_broadcast_fail_explicitly(self):
        with self.assertRaises(tk.UnsupportedGraphError):
            tk.compile()(lambda x: x + x).lower(mx.zeros((7,), dtype=mx.float16))
        with self.assertRaises(tk.UnsupportedGraphError):
            tk.compile()(lambda x, y: x + y).lower(mx.zeros((2, 3)), mx.zeros((3,)))

    def test_shape_specialization_is_reused(self):
        function = tk.compile()(lambda x: x + 2.0)
        first = function.lower(mx.zeros((7,)))
        self.assertIs(first, function.lower(mx.ones((7,))))
        self.assertIsNot(first, function.lower(mx.zeros((8,))))

    def test_invalid_schedule_and_backend_fail_explicitly(self):
        with self.assertRaises(tk.UnsupportedScheduleError):
            tk.Schedule(threads=0)
        with self.assertRaises(tk.UnsupportedScheduleError):
            tk.compile(backend="metal")

    def test_data_dependent_python_cannot_be_captured(self):
        def function(x):
            return x + 1 if x.item() > 0 else x - 1

        with self.assertRaises(ValueError):
            tk.compile()(function).lower(mx.array(1.0))


@unittest.skipUnless(mx.cuda.is_available(), "requires MLX CUDA")
class ExecutionTests(unittest.TestCase):
    def test_float_arithmetic_and_scalar_only_outputs(self):
        function = tk.compile()(lambda x, y: -mx.square(x) + x * y - 0.125)
        for shape in ((), (31,), (3, 171)):
            x = mx.random.normal(shape)
            y = mx.random.normal(shape)
            expected = -mx.square(x) + x * y - 0.125
            self.assertTrue(mx.allclose(function(x, y), expected, rtol=1e-5, atol=1e-6))

    def test_cuda_results_cover_partial_tiles_scalar_inputs_and_strides(self):
        for threads, elements in ((64, 1), (128, 4)):
            function = tk.compile(
                schedule=tk.Schedule(threads=threads, elements_per_thread=elements)
            )(lambda x, y: x * y + 2.0 - y)
            for size in (0, 1, 7, 127, 128, 129, 513):
                with self.subTest(threads=threads, elements=elements, size=size):
                    x = mx.arange(size, dtype=mx.float32)
                    y = mx.array(3.0)
                    self.assertTrue(mx.array_equal(function(x, y), x * y + 2.0 - y))
            x = mx.arange(42, dtype=mx.float32).reshape(6, 7).T
            self.assertTrue(mx.array_equal(function(x, mx.array(2.0)), x * 2.0))

    def test_cache_uses_current_input_values(self):
        function = tk.compile()(lambda x, y: -(x * x) + y)
        for offset in (0, 5):
            x = mx.arange(257, dtype=mx.float32) + offset
            y = mx.full((257,), 2.0)
            self.assertTrue(mx.array_equal(function(x, y), -(x * x) + y))


if __name__ == "__main__":
    unittest.main()
