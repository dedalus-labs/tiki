"""Warm host call plus blocking eval, with matched forward and full VJP work.

Invariant measured: the fast path is faster than the tree on forward and
backward at three shapes. Not a comparison with CUB or an optimized CuTe scan.
"""

import argparse
from collections.abc import Callable
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np

from affine_scan import Pair, affine_scan
from scan import associative_scan


def affine(left: Pair, right: Pair) -> Pair:
    al, bl = left
    ar, br = right
    return ar * al, ar * bl + br


def timed(function: Callable[..., Pair | list[mx.array]], *args: mx.array) -> float:
    for _ in range(5):
        mx.eval(function(*args))
    samples = []
    for _ in range(3):
        start = perf_counter()
        for _ in range(30):
            mx.eval(function(*args))
        samples.append((perf_counter() - start) * 1e6 / 30)
    return median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", choices=("eager", "compiled"), default="eager")
    args = parser.parse_args()

    def tree(a: mx.array, b: mx.array) -> Pair:
        return associative_scan(affine, (a, b), axis=1)

    def tree_vjp(
        a: mx.array, b: mx.array, gp: mx.array, gh: mx.array
    ) -> list[mx.array]:
        return mx.vjp(tree, (a, b), (gp, gh))[1]

    tree_forward = mx.compile(tree) if args.tree == "compiled" else tree
    tree_backward = mx.compile(tree_vjp) if args.tree == "compiled" else tree_vjp
    kernel = mx.compile(affine_scan)
    kernel_vjp = mx.compile(
        lambda a, b, gp, gh: mx.vjp(affine_scan, (a, b), (gp, gh))[1]
    )
    print(
        f"tree={args.tree}, timings include host dispatch and blocking eval, VJP includes required forward work"
    )
    rng = np.random.default_rng(19)
    for shape in ((32, 128), (256, 1024), (1024, 2048)):
        a = mx.array(rng.uniform(-0.8, 0.8, shape).astype(np.float32))
        b, gp, gh = (
            mx.array(rng.normal(size=shape).astype(np.float32)) for _ in range(3)
        )
        mx.eval(a, b, gp, gh)
        print(
            f"{shape}: tree fwd {timed(tree_forward, a, b):.1f} us | kernel fwd {timed(kernel, a, b):.1f} us | "
            f"tree VJP {timed(tree_backward, a, b, gp, gh):.1f} us | kernel VJP {timed(kernel_vjp, a, b, gp, gh):.1f} us"
        )


if __name__ == "__main__":
    main()
