"""Run native program contracts under tiki.launch with a real two-rank group."""

import os
import unittest
from functools import partial
from typing import Never

import tiki as tk

import compiler
from tiki_compiler.lowered import Lowered
from tiki_compiler.program import Program


def unexpected_kernel(lowered: Lowered, inputs: tuple[tk.array, ...], stream: tk.Stream) -> Never:
    raise AssertionError("native-only programs must not invoke the CuTe compiler")


@unittest.skipUnless("TIKI_RANK" in os.environ, "requires tiki.launch")
class NativeProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.group = tk.distributed.init(strict=True, backend=os.environ["TIKI_TEST_BACKEND"])
        if cls.group.size() != 2:
            raise RuntimeError("native program tests require exactly two ranks")

    def test_collectives_keep_their_operation_and_current_inputs(self) -> None:
        group = self.group
        ramp = tk.arange(32, dtype=tk.float32).reshape(8, 4)
        cases = (
            (tk.distributed.all_sum, 2 * ramp + 100),
            (tk.distributed.all_min, ramp),
            (tk.distributed.all_max, ramp + 100),
            (tk.distributed.all_gather, tk.concatenate([ramp, ramp + 100])),
        )
        for operation, expected in cases:
            function = compiler.compile()(partial(operation, group=group))
            x = ramp + 100 * group.rank()
            plan = function.lower(x)
            assert isinstance(plan, Program)
            (result,) = plan.launch((x,), unexpected_kernel)
            self.assertTrue(tk.array_equal(result, expected), operation.__name__)

    @unittest.skipUnless(os.environ.get("TIKI_TEST_BACKEND") == "nccl", "requires NCCL")
    def test_scatter_preserves_rank_partition(self) -> None:
        group = self.group
        ramp = tk.arange(32, dtype=tk.float32).reshape(8, 4)
        x = ramp + 100 * group.rank()
        function = compiler.compile()(lambda value: tk.distributed.sum_scatter(value, group=group))
        plan = function.lower(x)
        assert isinstance(plan, Program)
        (result,) = plan.launch((x,), unexpected_kernel)
        expected = (2 * ramp + 100)[group.rank() * 4 : (group.rank() + 1) * 4]
        self.assertTrue(tk.array_equal(result, expected))

    @unittest.skipUnless(os.environ.get("TIKI_TEST_BACKEND") == "nccl", "requires NCCL")
    def test_compiled_regions_preserve_collective_dependencies(self) -> None:
        group = self.group
        arch = tk.device_info(tk.gpu)["architecture"]
        assert isinstance(arch, str)
        schedule = compiler.Schedule(arch=arch)
        function = compiler.compile(schedule=schedule)(
            lambda x, weight: (
                x @ weight,
                tk.distributed.all_sum(x * 0.5, group=group) + 1.0,
            )
        )
        for offset in (0, 3):
            x = tk.full((32, 32), group.rank() + offset, dtype=tk.float32)
            weight = tk.ones((32, 16))
            product, reduced = function(x, weight)
            self.assertTrue(tk.array_equal(product, x @ weight))
            self.assertTrue(tk.all(reduced == 1.5 + offset).item())

    @unittest.skipUnless(os.environ.get("TIKI_TEST_BACKEND") == "nccl", "requires NCCL")
    def test_gather_preserves_distinct_communicators(self) -> None:
        group = self.group
        reverse = group.split(0, 1 - group.rank())
        function = compiler.compile()(
            lambda x: (
                tk.distributed.all_gather(x, group=group),
                tk.distributed.all_gather(x, group=reverse),
            )
        )
        x = tk.full((3,), group.rank(), dtype=tk.float32)
        plan = function.lower(x)
        assert isinstance(plan, Program)
        first, second = plan.launch((x,), unexpected_kernel)
        self.assertTrue(tk.array_equal(first, tk.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])))
        self.assertTrue(tk.array_equal(second, tk.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])))

    def test_collectives_on_separate_streams_keep_rank_agreement(self) -> None:
        group = self.group
        first, second = (tk.new_stream(tk.default_device()) for _ in range(2))
        function = compiler.compile()(
            lambda x, y: (
                tk.distributed.all_sum(x, group=group, stream=first),
                tk.distributed.all_max(y, group=group, stream=second),
            )
        )
        for offset in (0, 7):
            x = tk.full((31,), group.rank() + offset, dtype=tk.float32)
            y = tk.full((257,), 10 * group.rank() + offset, dtype=tk.float32)
            plan = function.lower(x, y)
            assert isinstance(plan, Program)
            total, largest = plan.launch((x, y), unexpected_kernel)
            self.assertTrue(tk.all(total == 1 + 2 * offset).item())
            self.assertTrue(tk.all(largest == 10 + offset).item())


if __name__ == "__main__":
    unittest.main()
