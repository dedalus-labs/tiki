"""Warm host-call plus eval timings of the affine scan: codegen kernels, the
hand-written CUDA kernel, and the generic tree. Run on the GH200."""

import sys
import time
from pathlib import Path

import mlx.core as mx

from associative_scan import ScanSchedule, associative_scan

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "associative_scan"))
from affine_scan import affine_scan  # noqa: E402
from scan import associative_scan as tree_scan  # noqa: E402

SHAPES = ((32, 128), (256, 1024), (1024, 2048))
SCHEDULE = ScanSchedule(threads=128, elements_per_thread=4)


def affine(left, right):
    return (right[0] * left[0], right[0] * left[1] + right[1])


def microseconds(function, repeats=20):
    for _ in range(3):
        mx.eval(function())
    start = time.perf_counter()
    for _ in range(repeats):
        mx.eval(function())
    return (time.perf_counter() - start) / repeats * 1e6


def main() -> None:
    print(
        "| shape | tree fwd | kernel fwd | codegen fwd | tree bwd | kernel bwd | codegen bwd |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for shape in SHAPES:
        a = mx.random.uniform(0.5, 1.0, shape)
        b = mx.random.normal(shape)
        mx.eval(a, b)
        forwards = {
            "tree": lambda: tree_scan(affine, (a, b), axis=1)[1],
            "kernel": lambda: affine_scan(a, b)[1],
            "codegen": lambda: associative_scan(
                affine, (a, b), axis=1, schedule=SCHEDULE
            )[1],
        }
        losses = {
            "tree": lambda a, b: tree_scan(affine, (a, b), axis=1)[1].sum(),
            "kernel": lambda a, b: affine_scan(a, b)[1].sum(),
            "codegen": lambda a, b: associative_scan(
                affine, (a, b), axis=1, schedule=SCHEDULE
            )[1].sum(),
        }
        row = [f"{shape[0]}x{shape[1]}"]
        row += [
            f"{microseconds(forwards[name]):.0f}"
            for name in ("tree", "kernel", "codegen")
        ]
        row += [
            f"{microseconds(lambda name=name: mx.grad(losses[name], argnums=(0, 1))(a, b)):.0f}"
            for name in ("tree", "kernel", "codegen")
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
