"""Run compiled arithmetic alongside matmul and NCCL on two NVIDIA GPUs.

From this checkout, with its CUDA build and CuTe DSL installed:
    PYTHONPATH=python:experiments/cute_backend tiki.launch \
        --backend nccl -n 2 -- python experiments/cute_backend/demo_overlap.py
"""

import tiki as tk

import compiler


def main() -> None:
    if not tk.cuda.is_available():
        raise compiler.BackendUnavailableError("this example requires two NVIDIA GPUs")
    group = tk.distributed.init(strict=True, backend="nccl")
    if group.size() < 2:
        raise RuntimeError("launch this example with --backend nccl -n 2")
    arch = tk.device_info(tk.gpu)["architecture"]
    assert isinstance(arch, str)
    schedule = compiler.Schedule(arch=arch)

    @compiler.compile(schedule=schedule)
    def step(a: tk.array, b: tk.array, x: tk.array) -> tuple[tk.array, tk.array]:
        return a @ b, tk.distributed.all_sum(x * 0.5, group=group) + 1.0

    a = tk.full((1024, 1024), group.rank() + 1, dtype=tk.float32)
    b = tk.full((1024, 1024), 0.25, dtype=tk.float32)
    x = tk.full((1024, 1024), group.rank() + 1, dtype=tk.float32)
    product, reduced = step(a, b, x)
    tk.eval(product, reduced)
    assert tk.all(product == 256 * (group.rank() + 1)).item()
    assert tk.all(reduced == group.size() * (group.size() + 1) / 4 + 1).item()
    print(f"rank {group.rank()}: matmul and compiled collective passed", flush=True)


if __name__ == "__main__":
    main()
