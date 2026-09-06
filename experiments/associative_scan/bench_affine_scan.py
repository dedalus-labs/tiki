"""Warm host call plus blocking eval, kernel versus compiled generic tree.

Invariant measured: the fast path is faster than the tree on forward and
backward at three shapes. Not a comparison with CUB or an optimized CuTe scan.
"""

from collections.abc import Callable
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np

from affine_scan import affine_scan, backward
from scan import associative_scan


def affine(left, right):
    al, bl = left
    ar, br = right
    return ar * al, ar * bl + br


def timed(function: Callable[..., object], *args: mx.array) -> float:
    for _ in range(5):
        mx.eval(function(*args))
    samples = []
    for _ in range(3):
        start = perf_counter()
        for _ in range(30):
            mx.eval(function(*args))
        samples.append((perf_counter() - start) * 1e6 / 30)
    return median(samples)


tree = mx.compile(lambda a, b: associative_scan(affine, (a, b), axis=1))
kernel = mx.compile(affine_scan)
tree_vjp = mx.compile(
    lambda a, b, gp, gh: mx.vjp(
        lambda a, b: associative_scan(affine, (a, b), axis=1), (a, b), (gp, gh)
    )[1]
)
kernel_bwd = mx.compile(lambda a, b, p, h, gp, gh: backward((a, b), (gp, gh), (p, h)))

rng = np.random.default_rng(19)
for shape in ((32, 128), (256, 1024), (1024, 2048)):
    a = mx.array(rng.uniform(-0.8, 0.8, shape).astype(np.float32))
    b, gp, gh = (mx.array(rng.normal(size=shape).astype(np.float32)) for _ in range(3))
    p, h = affine_scan(a, b)
    mx.eval(a, b, p, h, gp, gh)
    print(
        f"{shape}: tree fwd {timed(tree, a, b):.1f} us | kernel fwd {timed(kernel, a, b):.1f} us | "
        f"tree bwd {timed(tree_vjp, a, b, gp, gh):.1f} us | kernel bwd {timed(kernel_bwd, a, b, p, h, gp, gh):.1f} us"
    )
