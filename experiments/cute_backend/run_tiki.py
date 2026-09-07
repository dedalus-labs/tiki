"""Compile a CuTe kernel and launch its cubin through the Tiki CUDA runtime."""

from __future__ import annotations

import argparse
import json

import tiki as tk

from probe import THREADS, compile_reference_cubin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()

    if not tk.cuda.is_available():
        raise RuntimeError("Tiki CUDA backend is required")
    if args.size <= 0:
        raise ValueError("size must be positive")

    cubin, kernel_name = compile_reference_cubin(
        size=args.size,
        arch=args.arch,
    )
    left = tk.arange(args.size, dtype=tk.float32)
    right = tk.full((args.size,), 7, dtype=tk.float32)
    output = tk.fast.precompiled_cuda_kernel(
        name=kernel_name,
        compiled_source=cubin,
        inputs=[left, right],
        output_shapes=[left.shape],
        output_dtypes=[left.dtype],
        scalars=[],
        grid=(args.size, 1, 1),
        threadgroup=(THREADS, 1, 1),
    )[0]
    expected = left + right
    tk.eval(output, expected)

    max_error = tk.max(tk.abs(output - expected)).item()
    if max_error != 0:
        raise AssertionError(f"CuTe/Tiki result error: {max_error}")

    print(
        json.dumps(
            {
                "arch": args.arch,
                "size": args.size,
                "kernel": kernel_name,
                "cubin_bytes": len(cubin),
                "max_error": max_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
