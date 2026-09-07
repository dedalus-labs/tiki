# Copyright © 2023-2024 Apple Inc.

import argparse

import tiki as tk
import torch
from time_utils import measure_runtime


def benchmark_scatter_tiki(dst_shape, x_shape, idx_shapes):
    def scatter(dst, x, idx):
        dst[tuple(idx)] = x
        tk.eval(dst)

    idx = []
    for idx_shape in idx_shapes:
        idx.append(tk.random.randint(0, dst_shape[0] - 1, idx_shape))
    x = tk.random.normal(x_shape).astype(tk.float32)
    dst = tk.random.normal(dst_shape).astype(tk.float32)

    runtime = measure_runtime(scatter, dst=dst, x=x, idx=idx)
    print(f"Tiki: {runtime:.3f}ms")


def benchmark_scatter_torch(dst_shape, x_shape, idx_shapes, device):
    def scatter(dst, x, idx, device):
        dst[tuple(idx)] = x
        if device == torch.device("mps"):
            torch.mps.synchronize()

    idx = []
    for idx_shape in idx_shapes:
        idx.append(torch.randint(0, dst_shape[0] - 1, idx_shape).to(device))
    x = torch.randn(x_shape, dtype=torch.float32).to(device)
    dst = torch.randn(dst_shape, dtype=torch.float32).to(device)

    runtime = measure_runtime(scatter, dst=dst, x=x, idx=idx, device=device)
    print(f"PyTorch: {runtime:.3f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gather benchmarks.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU.")
    args = parser.parse_args()

    if args.cpu:
        tk.set_default_device(tk.cpu)
        device = torch.device("cpu")
    else:
        device = torch.device("mps")

    dst_shapes = [
        (10, 64),
        (100_000, 64),
        (1_000_000, 64),
        (100_000,),
        (200_000,),
        (20_000_000,),
        (10000, 64),
        (100, 64),
        (100, 10_000, 64),
        (10, 100, 100, 21),
        (1_000, 1_000, 10),
    ]
    idx_shapes = [
        [(1_000_000,)],
        [(1_000_000,)],
        [(100_000,)],
        [(1_000_000,)],
        [(20_000_000,)],
        [(20_000_000,)],
        [(1000000,)],
        [(10000000,)],
        [(1_000,)],
        [(10_000,)],
        [(1_000,), (1_000,)],
    ]
    x_shapes = [
        (1_000_000, 64),
        (1_000_000, 64),
        (100_000, 64),
        (1_000_000,),
        (20_000_000,),
        (20_000_000,),
        (1000000, 64),
        (10000000, 64),
        (1_000, 10_000, 64),
        (10_000, 100, 100, 21),
        (1_000, 10),
    ]

    for dst_shape, x_shape, idx_shape in zip(dst_shapes, x_shapes, idx_shapes):
        print("=" * 20)
        print(f"Dst: {dst_shape}, X {x_shape}, Indices {idx_shape}")
        benchmark_scatter_tiki(dst_shape, x_shape, idx_shape)
        benchmark_scatter_torch(dst_shape, x_shape, idx_shape, device=device)
