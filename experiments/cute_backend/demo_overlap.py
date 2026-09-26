"""Run compiled arithmetic alongside matmul and NCCL on two NVIDIA GPUs.

From this checkout, with its CUDA build and CuTe DSL installed:
    PYTHONPATH=python:experiments/cute_backend mlx.launch \
        --backend nccl -n 2 -- python experiments/cute_backend/demo_overlap.py
"""

import mlx.core as mx

import tiki as tk


def main() -> None:
    if not mx.cuda.is_available():
        raise tk.BackendUnavailableError("this example requires two NVIDIA GPUs")
    group = mx.distributed.init(strict=True, backend="nccl")
    if group.size() < 2:
        raise RuntimeError("launch this example with --backend nccl -n 2")
    schedule = tk.Schedule(arch=mx.device_info(mx.gpu)["architecture"])

    @tk.compile(schedule=schedule)
    def step(a, b, x):
        return a @ b, mx.distributed.all_sum(x * 0.5, group=group) + 1.0

    a = mx.full((1024, 1024), group.rank() + 1, dtype=mx.float32)
    b = mx.full((1024, 1024), 0.25, dtype=mx.float32)
    x = mx.full((1024, 1024), group.rank() + 1, dtype=mx.float32)
    product, reduced = step(a, b, x)
    mx.eval(product, reduced)
    assert mx.all(product == 256 * (group.rank() + 1)).item()
    assert mx.all(reduced == group.size() * (group.size() + 1) / 4 + 1).item()
    print(f"rank {group.rank()}: matmul and compiled collective passed", flush=True)


if __name__ == "__main__":
    main()
