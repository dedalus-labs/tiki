# Copyright © 2023-2024 Apple Inc.

import argparse

import tiki as tk
import torch
from time_utils import measure_runtime


def benchmark_gather_tiki(x_shape, idx_shape):
    def gather(x, idx):
        tk.eval(x[idx])

    idx = tk.random.randint(0, x_shape[0] - 1, idx_shape)
    x = tk.random.normal(x_shape).astype(tk.float32)

    runtime = measure_runtime(gather, x=x, idx=idx)
    print(f"Tiki: {runtime:.3f}ms")


def benchmark_gather_torch(x_shape, idx_shape, device):
    def gather(x, idx, device):
        _ = x[idx]
        if device == torch.device("mps"):
            torch.mps.synchronize()

    idx = torch.randint(0, x_shape[0] - 1, idx_shape).to(device)
    x = torch.randn(x_shape, dtype=torch.float32).to(device)

    runtime = measure_runtime(gather, x=x, idx=idx, device=device)
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

    idx_shapes = [(1_000_000,), (100_000,), ()]
    x_shapes = [(100, 64), (100, 1024), (4, 1_000_000)]

    for x_shape, idx_shape in zip(x_shapes, idx_shapes):
        print("=" * 20)
        print(f"X {x_shape}, Indices {idx_shape}")
        benchmark_gather_tiki(x_shape, idx_shape)
        benchmark_gather_torch(x_shape, idx_shape, device=device)
