"""Contracts for mixing native matrix operations with generated CuTe regions."""

import unittest
from typing import Never

import tiki as tk

import compiler
from tiki_compiler.arrays import single
from tiki_compiler.lowered import Lowered
from tiki_compiler.program import Kernel, Program


class ProgramTests(unittest.TestCase):
    def test_arithmetic_keeps_explicit_stream_placement(self) -> None:
        stream = tk.new_stream(tk.default_device())
        function = compiler.compile()(lambda x: tk.multiply(x, 2.0, stream=stream))
        lowered = function.lower(tk.ones((7,)))
        assert isinstance(lowered, Program)
        self.assertEqual(len(lowered.stages), 1)
        self.assertEqual(lowered.stages[0].stream, stream)

    def test_native_only_plan_executes_with_current_inputs(self) -> None:
        function = compiler.compile()(lambda x, weight: x @ weight)
        for offset in (0, 3):
            x = tk.arange(32, dtype=tk.float32).reshape(4, 8) + offset
            weight = tk.ones((8, 3))
            lowered = function.lower(x, weight)
            assert isinstance(lowered, Program)

            def unexpected_kernel(
                lowered: Lowered, inputs: tuple[tk.array, ...], stream: tk.Stream
            ) -> Never:
                self.fail("a native-only program must not launch a CuTe region")

            (result,) = lowered.launch((x, weight), unexpected_kernel)
            self.assertTrue(tk.array_equal(result, x @ weight))

    def test_default_stream_is_part_of_specialization(self) -> None:
        function = compiler.compile()(lambda x, weight: x @ weight)
        inputs = (tk.zeros((4, 8)), tk.zeros((8, 3)))
        first = function.lower(*inputs)
        assert isinstance(first, Program)
        self.assertIs(first, function.lower(*inputs))
        other = tk.new_stream(tk.default_device())
        with tk.stream(other):
            second = function.lower(*inputs)
            assert isinstance(second, Program)
        self.assertIsNot(first, second)
        self.assertEqual(second.stages[0].stream, other)

    def test_fusion_preserves_explicit_stream_boundaries(self) -> None:
        first, second = (tk.new_stream(tk.default_device()) for _ in range(2))

        def operation(x: tk.array, weight: tk.array) -> tk.array:
            product = tk.matmul(x, weight, stream=first)
            scaled = tk.multiply(product, 2.0, stream=first)
            return tk.add(scaled, 1.0, stream=second)

        lowered = compiler.compile()(operation).lower(tk.zeros((4, 8)), tk.zeros((8, 3)))
        assert isinstance(lowered, Program)
        self.assertEqual([stage.stream for stage in lowered.stages], [first, first, second])

    def test_lowering_keeps_matrix_operations_between_fused_regions(self) -> None:
        function = compiler.compile()(lambda x, weight: (x * 2.0) @ weight + 1.0)
        lowered = function.lower(tk.zeros((4, 8)), tk.zeros((8, 3)))
        assert isinstance(lowered, Program)
        self.assertEqual([stage.operation for stage in lowered.stages], ["CuTe", "Matmul", "CuTe"])
        assert isinstance(lowered.stages[0], Kernel)
        assert isinstance(lowered.stages[-1], Kernel)
        self.assertIn("arith.mulf", lowered.stages[0].lowered.mlir)
        self.assertIn("arith.addf", lowered.stages[-1].lowered.mlir)
        self.assertEqual(lowered.output_shapes, ((4, 3),))

    def test_lowering_preserves_independent_outputs(self) -> None:
        function = compiler.compile()(lambda x, weight, other: (x @ weight, other * 2.0 + 1.0))
        lowered = function.lower(tk.zeros((4, 8)), tk.zeros((8, 3)), tk.zeros((11,)))
        assert isinstance(lowered, Program)
        self.assertEqual(lowered.output_shapes, ((4, 3), (11,)))
        self.assertEqual([stage.operation for stage in lowered.stages], ["Matmul", "CuTe"])

    def test_unsupported_operations_remain_errors_inside_programs(self) -> None:
        with self.assertRaisesRegex(compiler.UnsupportedGraphError, "Exp"):
            compiler.compile()(lambda x, weight: tk.exp(x @ weight)).lower(
                tk.zeros((4, 8)), tk.zeros((8, 3))
            )


@unittest.skipUnless(tk.cuda.is_available(), "requires Tiki CUDA")
class ProgramExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        arch = tk.device_info(tk.gpu)["architecture"]
        assert isinstance(arch, str)
        self.schedule = compiler.Schedule(arch=arch)

    def test_target_preserves_cooperative_schedules(self) -> None:
        def rms(x: tk.array) -> tk.array:
            return x * tk.rsqrt(tk.mean(x * x, axis=-1, keepdims=True) + 1e-6)

        x = tk.arange(195, dtype=tk.float32).reshape(3, 65)
        normalized = compiler.compile(schedule=compiler.RowSchedule(arch=self.schedule.arch))(rms)
        transposed = compiler.compile(schedule=compiler.TransposeSchedule(arch=self.schedule.arch))(
            lambda value: value.T
        )
        self.assertTrue(tk.allclose(single(normalized(x)), rms(x), atol=2e-6, rtol=2e-5))
        self.assertTrue(tk.array_equal(single(transposed(x)), x.T))

    def test_fused_regions_preserve_strided_matrix_inputs(self) -> None:
        function = compiler.compile(schedule=self.schedule)(lambda x, weight: (x * 2.0) @ weight + 1.0)
        weight = tk.arange(24, dtype=tk.float32).reshape(8, 3)
        for offset in (0, 3):
            views = (
                (tk.arange(60, dtype=tk.float32) + offset).reshape(5, 12)[:, 1:9],
                (tk.arange(40, dtype=tk.float32) + offset).reshape(8, 5).T,
            )
            for x in views:
                self.assertTrue(tk.array_equal(single(function(x, weight)), (x * 2) @ weight + 1))

    def test_independent_outputs_keep_shapes_and_partial_tiles(self) -> None:
        function = compiler.compile(schedule=self.schedule)(
            lambda x, weight, other: (x @ weight, other * 2.0 + 1.0)
        )
        x, weight = tk.ones((4, 8)), tk.ones((8, 3))
        for count in (0, 1, 31, 257):
            other = tk.arange(count, dtype=tk.float32)
            product, affine = function(x, weight, other)
            self.assertTrue(tk.array_equal(product, x @ weight))
            self.assertTrue(tk.array_equal(affine, other * 2 + 1))

    def test_split_stream_dependencies_survive_repeated_execution(self) -> None:
        compute, output = tk.new_stream(tk.gpu), tk.new_stream(tk.gpu)

        def operation(x: tk.array, weight: tk.array) -> tk.array:
            product = tk.matmul(x, weight, stream=compute)
            return tk.add(product, 1.0, stream=output)

        function = compiler.compile(schedule=self.schedule)(operation)
        for offset in range(5):
            x = tk.full((4, 8), offset, dtype=tk.float32)
            weight = tk.ones((8, 3))
            self.assertTrue(tk.array_equal(single(function(x, weight)), x @ weight + 1))


if __name__ == "__main__":
    unittest.main()
