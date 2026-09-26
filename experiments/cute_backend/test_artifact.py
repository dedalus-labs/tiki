"""Verify the pinned compiler binding and recursive execution on the selected GPU."""

import unittest

from mlx import core

import tiki
from associative_scan import ScanSchedule, associative_scan
from tiki_compiler.arrays import single
from tiki_compiler.artifact import binary
from tiki_compiler.lowered import Lowered


@unittest.skipUnless(core.cuda.is_available(), "requires MLX CUDA")
class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        arch = core.device_info(core.gpu)["architecture"]
        assert isinstance(arch, str)
        self.arch = arch

    def test_compiler_binding_produces_a_reusable_executable(self) -> None:
        function = tiki.compile(schedule=tiki.Schedule(arch=self.arch))(lambda value: value * 2.0)
        values = core.arange(257, dtype=core.float32)
        lowered = function.lower(values)
        assert isinstance(lowered, Lowered)
        artifact = binary(function.io, lowered)
        self.assertTrue(artifact.cubin.startswith(b"\x7fELF"))
        self.assertIn(f".target {self.arch}", artifact.ptx)
        self.assertIs(artifact, binary(function.io, lowered))
        self.assertTrue(core.array_equal(single(function(values)), values * 2.0))
        with self.assertRaisesRegex(ValueError, "CustomKernel"):
            core.jvp(function.launch, [values], [core.ones_like(values)])

    def test_scan_derivatives_use_the_selected_target(self) -> None:
        schedule = ScanSchedule(arch=self.arch)

        def cumulative(values: core.array) -> core.array:
            result = associative_scan(core.add, values, axis=1, schedule=schedule)
            return result

        def total(values: core.array) -> core.array:
            result = cumulative(values).sum()
            return result

        values = core.ones((2, 577), dtype=core.float32)
        self.assertTrue(core.array_equal(cumulative(values), core.cumsum(values, axis=1)))
        expected = core.cumsum(values, axis=1, reverse=True)
        self.assertTrue(core.array_equal(core.grad(total)(values), expected))
        tangent = core.jvp(cumulative, [values], [values])[1][0]
        self.assertTrue(core.array_equal(tangent, core.cumsum(values, axis=1)))


if __name__ == "__main__":
    unittest.main()
