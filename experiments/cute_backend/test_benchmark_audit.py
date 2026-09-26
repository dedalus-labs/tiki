"""Contracts for benchmark validation and CUDA resource ownership."""

import ctypes
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np


def load_benchmark():
    mlx = Mock()
    cuda = Mock()
    cutlass = Mock()
    modules = {
        "mlx": mlx,
        "mlx.core": mlx.core,
        "cuda": cuda,
        "cuda.bindings": cuda.bindings,
        "cutlass": cutlass,
        "tiki": Mock(),
        "demo_cooperative": Mock(),
    }
    spec = importlib.util.spec_from_file_location(
        "benchmark_audit", Path(__file__).with_name("benchmark_audit.py")
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(module)
    return module


class ValidationTests(unittest.TestCase):
    def test_each_variant_must_write_every_output(self):
        # Each variant must write all outputs. Stale results hide missing writes.
        for written in (0, 1, 4):
            with self.subTest(written=written):
                benchmark = load_benchmark()
                cuda = benchmark.cuda
                success = cuda.CUresult.CUDA_SUCCESS
                expected = np.arange(4, dtype=np.float32)
                device = np.empty_like(expected)
                cuda.cuModuleLoadData.return_value = (success, 1)
                cuda.cuModuleGetFunction.return_value = (success, 2)
                cuda.cuModuleUnload.return_value = (success,)
                cuda.cuStreamSynchronize.return_value = (success,)
                benchmark.testing.benchmark.return_value = 1.0

                def launch(*args):
                    device[:count] = expected[:count]
                    return (success,)

                def copy_to_host(destination, source, size):
                    ctypes.memmove(destination, device.ctypes.data, size)
                    return (success,)

                def copy_to_device(destination, source, size):
                    ctypes.memmove(device.ctypes.data, source, size)
                    return (success,)

                cuda.cuLaunchKernel.side_effect = launch
                cuda.cuMemcpyDtoH.side_effect = copy_to_host
                cuda.cuMemcpyHtoD.side_effect = copy_to_device
                args = (b"cubin", "kernel", ([1, 2, 3], 0, 1, 32, 0), expected)
                count = expected.size
                self.assertEqual(benchmark.measure(*args)["max_error"], 0.0)
                count = written
                if written < expected.size:
                    with self.assertRaises(AssertionError):
                        benchmark.measure(*args)
                else:
                    self.assertEqual(benchmark.measure(*args)["max_error"], 0.0)


class AllocationTests(unittest.TestCase):
    def test_all_acquired_buffers_are_freed_when_allocation_fails(self):
        # Every acquired buffer must be freed. Later allocation failures expose leaks.
        for failed in (1, 2, 3):
            with self.subTest(failed=failed), TemporaryDirectory() as directory:
                benchmark = load_benchmark()
                cuda = benchmark.cuda
                success = cuda.CUresult.CUDA_SUCCESS
                cuda.cuMemAlloc.side_effect = [
                    *((success, pointer) for pointer in range(1, failed)),
                    ("CUDA_ERROR_OUT_OF_MEMORY",),
                ]
                cuda.cuMemFree.return_value = (success,)
                lowered = (
                    benchmark.tk.compile.return_value.return_value.lower.return_value
                )
                lowered.mlir = "mlir"
                benchmark.tk.binary.return_value.ptx = "ptx"
                benchmark.compile_reference = Mock(return_value=b"cubin")
                schedule = Mock(threads_per_row=32, rows_per_block=4)
                with self.assertRaisesRegex(RuntimeError, "CUDA_ERROR_OUT_OF_MEMORY"):
                    benchmark.benchmark_case(
                        (2, 4), schedule, 0, Path(directory), False
                    )
                self.assertEqual(
                    [call.args[0] for call in cuda.cuMemFree.call_args_list],
                    list(range(1, failed)),
                )


if __name__ == "__main__":
    unittest.main()
