# Copyright © 2023-2024 Apple Inc.

import tiki as tk
import tiki.nn as nn
from time_utils import time_fn


def rms_norm(x, w, eps):
    ot = x.dtype
    x = x.astype(tk.float32)
    n = tk.rsqrt(x.square().mean(-1, keepdims=True) + eps)
    y = (x * n).astype(ot)
    if w is not None:
        y = y * w
    return y


def time_rms_norm():
    f1 = lambda x, w, y: (rms_norm(x, w, 1e-5) * y).sum()
    f2 = lambda x, w, y: (tk.fast.rms_norm(x, w, 1e-5) * y).sum()
    g1 = tk.grad(f1, argnums=(0, 1))
    g2 = tk.grad(f2, argnums=(0, 1))

    x = tk.random.uniform(shape=(8, 1024, 4096)).astype(tk.float16)
    w = tk.random.uniform(shape=(4096,)).astype(tk.float16)
    y = tk.random.uniform(shape=(8, 1024, 4096)).astype(tk.float16)
    tk.eval(x, w, y)

    def rms_norm_loop(g, x, w):
        gx, gw = x, w
        for _ in range(32):
            gx, gw = g(gx, gw, y)
        return gx, gw

    time_fn(rms_norm_loop, g1, x, w)
    time_fn(rms_norm_loop, g2, x, w)
    time_fn(rms_norm_loop, tk.compile(g1), x, w)
    time_fn(rms_norm_loop, tk.compile(g2), x, w)

    f1 = lambda x, y: (rms_norm(x, None, 1e-5) * y).sum()
    f2 = lambda x, y: (tk.fast.rms_norm(x, None, 1e-5) * y).sum()
    g1 = tk.grad(f1, argnums=(0,))
    g2 = tk.grad(f2, argnums=(0,))

    x = tk.random.uniform(shape=(8, 1024, 4096)).astype(tk.float16)
    w = tk.random.uniform(shape=(4096,)).astype(tk.float16)
    y = tk.random.uniform(shape=(8, 1024, 4096)).astype(tk.float16)
    tk.eval(x, w, y)

    def rms_norm_loop(g, x):
        gx = x
        for _ in range(32):
            gx = g(gx, y)
        return gx

    time_fn(rms_norm_loop, g1, x)
    time_fn(rms_norm_loop, g2, x)
    time_fn(rms_norm_loop, tk.compile(g1), x)
    time_fn(rms_norm_loop, tk.compile(g2), x)


if __name__ == "__main__":
    time_rms_norm()
