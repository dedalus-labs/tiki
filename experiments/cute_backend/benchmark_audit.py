"""Compare CuTe and NVCC device binaries under the same CUDA Graph harness."""

import argparse
import ctypes
import json
import statistics
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
from cuda.bindings import driver as cuda
from cutlass import testing
from demo_cooperative import rms_norm

import tiki as tk


def checked(result):
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA call failed: {result[0]}")
    return result[1:]


def compile_reference(
    shape: tuple[int, int], schedule: tk.RowSchedule, output: Path
) -> bytes:
    rows, width = shape
    subprocess.run(
        [
            "nvcc",
            "-cubin",
            "-O3",
            "--fmad=true",
            "-arch=sm_90",
            "-std=c++17",
            f"-DROWS={rows}",
            f"-DWIDTH={width}",
            f"-DROW_THREADS={schedule.threads_per_row}",
            f"-DBLOCK_ROWS={schedule.rows_per_block}",
            str(Path(__file__).with_name("benchmark_rms.cu")),
            "-o",
            str(output),
        ],
        check=True,
    )
    return output.read_bytes()


def benchmark_case(
    shape: tuple[int, int],
    schedule: tk.RowSchedule,
    stream,
    output: Path,
    reverse: bool,
):
    rows, width = shape
    lowered = tk.compile(schedule=schedule)(rms_norm).lower(
        mx.zeros(shape), mx.zeros((width,))
    )
    cute_binary = tk.binary(lowered)
    prefix = f"{rows}x{width}-t{schedule.threads_per_row}-r{schedule.rows_per_block}"
    (output / f"{prefix}.mlir").write_text(lowered.mlir)
    (output / f"{prefix}.ptx").write_text(cute_binary.ptx)
    cpp_path = output / f"{prefix}-nvcc.cubin"
    cpp_binary = compile_reference(shape, schedule, cpp_path)
    rng = np.random.default_rng(17)
    x = rng.uniform(-1, 1, size=shape).astype(np.float32)
    weight = rng.uniform(-1, 1, size=(width,)).astype(np.float32)
    expected = (
        x
        / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-6)
        * weight
    ).astype(np.float32)
    pointers = [
        checked(cuda.cuMemAlloc(size))[0]
        for size in (x.nbytes, weight.nbytes, x.nbytes)
    ]
    try:
        checked(cuda.cuMemcpyHtoD(pointers[0], x.ctypes.data, x.nbytes))
        checked(cuda.cuMemcpyHtoD(pointers[1], weight.ctypes.data, weight.nbytes))
        results = {}
        variants = (
            ("cute", cute_binary.cubin, "tiki_fused", lowered.shared_memory_bytes),
            ("nvcc", cpp_binary, "rms_reference", 0),
        )
        for name, cubin, kernel, shared in (variants[::-1] if reverse else variants):
            results[name] = measure(
                cubin,
                kernel,
                (
                    pointers,
                    stream,
                    lowered.grid[0] // schedule.threads,
                    schedule.threads,
                    shared,
                ),
                expected,
            )
        report = {
            "shape": shape,
            "threads_per_row": schedule.threads_per_row,
            "rows_per_block": schedule.rows_per_block,
            **results,
        }
        print(json.dumps(report), flush=True)
        return report
    finally:
        for pointer in pointers:
            checked(cuda.cuMemFree(pointer))


def measure(cubin, kernel: str, launch, expected: np.ndarray):
    pointers, stream, blocks, threads, shared = launch
    (module,) = checked(cuda.cuModuleLoadData(cubin))
    try:
        (function,) = checked(cuda.cuModuleGetFunction(module, kernel.encode()))
        values = [ctypes.c_uint64(int(pointer)) for pointer in pointers]
        params = (ctypes.c_void_p * len(values))(
            *(ctypes.addressof(value) for value in values)
        )

        def run():
            checked(
                cuda.cuLaunchKernel(
                    function,
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    shared,
                    stream,
                    ctypes.addressof(params),
                    0,
                )
            )

        run()
        checked(cuda.cuStreamSynchronize(stream))
        actual = np.empty_like(expected)
        checked(cuda.cuMemcpyDtoH(actual.ctypes.data, pointers[-1], actual.nbytes))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        samples = [
            testing.benchmark(
                run,
                kernel_arguments=testing.JitArguments(),
                stream=stream,
                use_cuda_graphs=True,
                warmup_iterations=20,
                iterations=200,
            )
            for _ in range(3)
        ]
        return {
            "median_us": statistics.median(samples),
            "samples_us": samples,
            "max_error": float(np.max(np.abs(actual - expected))),
        }
    finally:
        checked(cuda.cuModuleUnload(module))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subwarp", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checked(cuda.cuInit(0))
    (device,) = checked(cuda.cuDeviceGet(0))
    (context,) = checked(cuda.cuDevicePrimaryCtxRetain(device))
    checked(cuda.cuCtxSetCurrent(context))
    (stream,) = checked(cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING))
    try:
        reports = []
        for shape in ((8192, 16), (8192, 64), (2048, 4096)):
            configs = [(32, 4), (128, 1)]
            if args.subwarp and shape[1] <= 64:
                configs.extend([(8, 16), (16, 8)])
            for threads, rows in configs:
                reports.append(
                    benchmark_case(
                        shape,
                        tk.RowSchedule(threads_per_row=threads, rows_per_block=rows),
                        stream,
                        args.output,
                        args.reverse,
                    )
                )
        (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        checked(cuda.cuStreamDestroy(stream))
        checked(cuda.cuDevicePrimaryCtxRelease(device))


if __name__ == "__main__":
    main()
