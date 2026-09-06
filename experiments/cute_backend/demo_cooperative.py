"""Inspect native row reductions and swizzled transpose schedules."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import tiki as tk


def rms_norm(x: mx.array, weight: mx.array) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6) * weight


def transpose(x: mx.array) -> mx.array:
    return x.T


def save_case(
    name: str,
    function: tk.Compiled,
    inputs: tuple[mx.array, ...],
    output: Path,
    execute: bool,
) -> None:
    lowered = function.lower(*inputs)
    directory = output / name
    directory.mkdir()
    report = {
        "schedule": asdict(lowered.schedule),
        "grid_threads": lowered.grid,
        "shared_memory_bytes": lowered.shared_memory_bytes,
    }
    (directory / "kernel.mlir").write_text(lowered.mlir)
    (directory / "graph.json").write_text(
        json.dumps(asdict(lowered.graph), indent=2) + "\n"
    )
    if execute:
        artifact = tk.binary(lowered)
        (directory / "kernel.ptx").write_text(artifact.ptx)
        (directory / "kernel.cubin").write_bytes(artifact.cubin)
        actual = function(*inputs)
        expected = function.function(*inputs)
        if not mx.allclose(actual, expected, atol=2e-6, rtol=2e-5):
            raise AssertionError(f"incorrect output: {name}")
        report["max_error"] = mx.max(mx.abs(actual - expected)).item()
        report["cubin_bytes"] = len(artifact.cubin)
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": name, **report}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(17)
    x = mx.random.normal((5, 129))
    weight = mx.random.normal((129,))
    matrix = mx.arange(33 * 65, dtype=mx.float32).reshape(33, 65)
    for name, schedule in (
        ("rms_warp", tk.RowSchedule(threads_per_row=32, rows_per_block=4)),
        ("rms_block", tk.RowSchedule(threads_per_row=64, rows_per_block=2)),
    ):
        function = tk.compile(schedule=schedule)(rms_norm)
        save_case(name, function, (x, weight), args.output, args.execute)
    for name, swizzle in (
        ("transpose_plain", tk.Swizzle(bits=0)),
        ("transpose_swizzle", tk.Swizzle(bits=5)),
    ):
        function = tk.compile(schedule=tk.TransposeSchedule(swizzle=swizzle))(transpose)
        save_case(name, function, (matrix,), args.output, args.execute)


if __name__ == "__main__":
    main()
