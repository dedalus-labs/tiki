"""Run native program contracts under mlx.launch with a real two-rank group."""

import os
import unittest

import mlx.core as mx

import tiki as tk


def unexpected_kernel(*args):
    raise AssertionError("native-only programs must not invoke the CuTe compiler")


@unittest.skipUnless("MLX_RANK" in os.environ, "requires mlx.launch")
class NativeProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.group = mx.distributed.init(
            strict=True, backend=os.environ["TIKI_TEST_BACKEND"]
        )
        if cls.group.size() != 2:
            raise RuntimeError("native program tests require exactly two ranks")

    def test_collectives_keep_their_operation_and_current_inputs(self):
        group = self.group
        ramp = mx.arange(32, dtype=mx.float32).reshape(8, 4)
        cases = (
            (mx.distributed.all_sum, 2 * ramp + 100),
            (mx.distributed.all_min, ramp),
            (mx.distributed.all_max, ramp + 100),
            (mx.distributed.all_gather, mx.concatenate((ramp, ramp + 100))),
        )
        for operation, expected in cases:
            function = tk.compile()(lambda x: operation(x, group=group))
            x = ramp + 100 * group.rank()
            plan = function.lower(x)
            (result,) = plan.launch((x,), unexpected_kernel)
            self.assertTrue(mx.array_equal(result, expected), operation.__name__)

    @unittest.skipUnless(os.environ.get("TIKI_TEST_BACKEND") == "nccl", "requires NCCL")
    def test_scatter_preserves_rank_partition(self):
        group = self.group
        ramp = mx.arange(32, dtype=mx.float32).reshape(8, 4)
        x = ramp + 100 * group.rank()
        function = tk.compile()(
            lambda value: mx.distributed.sum_scatter(value, group=group)
        )
        (result,) = function.lower(x).launch((x,), unexpected_kernel)
        expected = (2 * ramp + 100)[group.rank() * 4 : (group.rank() + 1) * 4]
        self.assertTrue(mx.array_equal(result, expected))

    @unittest.skipUnless(os.environ.get("TIKI_TEST_BACKEND") == "nccl", "requires NCCL")
    def test_compiled_regions_preserve_collective_dependencies(self):
        group = self.group
        schedule = tk.Schedule(arch=mx.device_info(mx.gpu)["architecture"])
        function = tk.compile(schedule=schedule)(
            lambda x, weight: (
                x @ weight,
                mx.distributed.all_sum(x * 0.5, group=group) + 1.0,
            )
        )
        for offset in (0, 3):
            x = mx.full((32, 32), group.rank() + offset, dtype=mx.float32)
            weight = mx.ones((32, 16))
            product, reduced = function(x, weight)
            self.assertTrue(mx.array_equal(product, x @ weight))
            self.assertTrue(mx.all(reduced == 1.5 + offset).item())

    @unittest.skipUnless(os.environ.get("TIKI_TEST_BACKEND") == "nccl", "requires NCCL")
    def test_gather_preserves_distinct_communicators(self):
        group = self.group
        reverse = group.split(0, 1 - group.rank())
        function = tk.compile()(
            lambda x: (
                mx.distributed.all_gather(x, group=group),
                mx.distributed.all_gather(x, group=reverse),
            )
        )
        x = mx.full((3,), group.rank(), dtype=mx.float32)
        first, second = function.lower(x).launch((x,), unexpected_kernel)
        self.assertTrue(mx.array_equal(first, mx.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])))
        self.assertTrue(
            mx.array_equal(second, mx.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
        )

    def test_collectives_on_separate_streams_keep_rank_agreement(self):
        group = self.group
        first, second = (mx.new_stream(mx.default_device()) for _ in range(2))
        function = tk.compile()(
            lambda x, y: (
                mx.distributed.all_sum(x, group=group, stream=first),
                mx.distributed.all_max(y, group=group, stream=second),
            )
        )
        for offset in (0, 7):
            x = mx.full((31,), group.rank() + offset, dtype=mx.float32)
            y = mx.full((257,), 10 * group.rank() + offset, dtype=mx.float32)
            plan = function.lower(x, y)
            total, largest = plan.launch((x, y), unexpected_kernel)
            self.assertTrue(mx.all(total == 1 + 2 * offset).item())
            self.assertTrue(mx.all(largest == 10 + offset).item())


if __name__ == "__main__":
    unittest.main()
