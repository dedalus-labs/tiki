"""Contracts for mixing native matrix operations with generated CuTe regions."""

import unittest

import mlx.core as mx

import tiki as tk


class ProgramTests(unittest.TestCase):
    def test_arithmetic_keeps_explicit_stream_placement(self):
        stream = mx.new_stream(mx.default_device())
        function = tk.compile()(lambda x: mx.multiply(x, 2.0, stream=stream))
        lowered = function.lower(mx.ones((7,)))
        self.assertEqual(len(lowered.stages), 1)
        self.assertEqual(lowered.stages[0].stream, stream)

    def test_native_only_plan_executes_with_current_inputs(self):
        function = tk.compile()(lambda x, weight: x @ weight)
        for offset in (0, 3):
            x = mx.arange(32, dtype=mx.float32).reshape(4, 8) + offset
            weight = mx.ones((8, 3))
            lowered = function.lower(x, weight)

            def unexpected_kernel(*args):
                self.fail("a native-only program must not launch a CuTe region")

            (result,) = lowered.launch((x, weight), unexpected_kernel)
            self.assertTrue(mx.array_equal(result, x @ weight))

    def test_default_stream_is_part_of_specialization(self):
        function = tk.compile()(lambda x, weight: x @ weight)
        inputs = (mx.zeros((4, 8)), mx.zeros((8, 3)))
        first = function.lower(*inputs)
        self.assertIs(first, function.lower(*inputs))
        other = mx.new_stream(mx.default_device())
        with mx.stream(other):
            second = function.lower(*inputs)
        self.assertIsNot(first, second)
        self.assertEqual(second.stages[0].stream, other)

    def test_fusion_preserves_explicit_stream_boundaries(self):
        first, second = (mx.new_stream(mx.default_device()) for _ in range(2))

        def operation(x, weight):
            product = mx.matmul(x, weight, stream=first)
            scaled = mx.multiply(product, 2.0, stream=first)
            return mx.add(scaled, 1.0, stream=second)

        lowered = tk.compile()(operation).lower(mx.zeros((4, 8)), mx.zeros((8, 3)))
        self.assertEqual(
            [stage.stream for stage in lowered.stages], [first, first, second]
        )

    def test_lowering_keeps_matrix_operations_between_fused_regions(self):
        function = tk.compile()(lambda x, weight: (x * 2.0) @ weight + 1.0)
        lowered = function.lower(mx.zeros((4, 8)), mx.zeros((8, 3)))
        self.assertEqual(
            [stage.operation for stage in lowered.stages], ["CuTe", "Matmul", "CuTe"]
        )
        self.assertIn("arith.mulf", lowered.stages[0].lowered.mlir)
        self.assertIn("arith.addf", lowered.stages[-1].lowered.mlir)
        self.assertEqual(lowered.output_shapes, ((4, 3),))

    def test_lowering_preserves_independent_outputs(self):
        function = tk.compile()(
            lambda x, weight, other: (x @ weight, other * 2.0 + 1.0)
        )
        lowered = function.lower(mx.zeros((4, 8)), mx.zeros((8, 3)), mx.zeros((11,)))
        self.assertEqual(lowered.output_shapes, ((4, 3), (11,)))
        self.assertEqual(
            [stage.operation for stage in lowered.stages], ["Matmul", "CuTe"]
        )

    def test_unsupported_operations_remain_errors_inside_programs(self):
        with self.assertRaisesRegex(tk.UnsupportedGraphError, "Exp"):
            tk.compile()(lambda x, weight: mx.exp(x @ weight)).lower(
                mx.zeros((4, 8)), mx.zeros((8, 3))
            )


@unittest.skipUnless(mx.cuda.is_available(), "requires MLX CUDA")
class ProgramExecutionTests(unittest.TestCase):
    def setUp(self):
        self.schedule = tk.Schedule(arch=mx.device_info(mx.gpu)["architecture"])

    def test_target_preserves_cooperative_schedules(self):
        def rms(x):
            return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)

        x = mx.arange(195, dtype=mx.float32).reshape(3, 65)
        normalized = tk.compile(schedule=tk.RowSchedule(arch=self.schedule.arch))(rms)
        transposed = tk.compile(schedule=tk.TransposeSchedule(arch=self.schedule.arch))(
            lambda value: value.T
        )
        self.assertTrue(mx.allclose(normalized(x), rms(x), atol=2e-6, rtol=2e-5))
        self.assertTrue(mx.array_equal(transposed(x), x.T))

    def test_fused_regions_preserve_strided_matrix_inputs(self):
        function = tk.compile(schedule=self.schedule)(
            lambda x, weight: (x * 2.0) @ weight + 1.0
        )
        weight = mx.arange(24, dtype=mx.float32).reshape(8, 3)
        for offset in (0, 3):
            views = (
                (mx.arange(60, dtype=mx.float32) + offset).reshape(5, 12)[:, 1:9],
                (mx.arange(40, dtype=mx.float32) + offset).reshape(8, 5).T,
            )
            for x in views:
                self.assertTrue(
                    mx.array_equal(function(x, weight), (x * 2) @ weight + 1)
                )

    def test_independent_outputs_keep_shapes_and_partial_tiles(self):
        function = tk.compile(schedule=self.schedule)(
            lambda x, weight, other: (x @ weight, other * 2.0 + 1.0)
        )
        x, weight = mx.ones((4, 8)), mx.ones((8, 3))
        for count in (0, 1, 31, 257):
            other = mx.arange(count, dtype=mx.float32)
            product, affine = function(x, weight, other)
            self.assertTrue(mx.array_equal(product, x @ weight))
            self.assertTrue(mx.array_equal(affine, other * 2 + 1))

    def test_split_stream_dependencies_survive_repeated_execution(self):
        compute, output = mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)

        def operation(x, weight):
            product = mx.matmul(x, weight, stream=compute)
            return mx.add(product, 1.0, stream=output)

        function = tk.compile(schedule=self.schedule)(operation)
        for offset in range(5):
            x = mx.full((4, 8), offset, dtype=mx.float32)
            weight = mx.ones((8, 3))
            self.assertTrue(mx.array_equal(function(x, weight), x @ weight + 1))


if __name__ == "__main__":
    unittest.main()
