# Copyright © 2023-2024 Apple Inc.

import argparse
import math
import random

import tiki as tk
from time_utils import time_fn


def bench_gelu():
    def gelu(x):
        return x * (1 + tk.erf(x / math.sqrt(2))) / 2

    x = tk.random.uniform(shape=(1000, 1024))

    def gen_fun(fun):
        def bench_fun(x):
            for _ in range(10):
                x = fun(x)
            return x

        return bench_fun

    time_fn(gen_fun(gelu), x, msg="fixed gelu")
    time_fn(gen_fun(tk.compile(gelu)), x, msg="compiled fixed gelu")

    def randint():
        return random.randint(1, x.shape[0])

    def gen_fun(fun):
        def bench_fun(x, y):
            x = x[: randint()]
            for _ in range(10):
                x = fun(x)
                y = fun(y)
            return x, y

        return bench_fun

    y = tk.random.uniform(shape=(1000, 1024))
    time_fn(gen_fun(gelu), x, y, msg="variable gelu")
    time_fn(gen_fun(tk.compile(gelu)), x, y, msg="compiled variable gelu")
    time_fn(
        gen_fun(tk.compile(gelu, shapeless=True)),
        x,
        y,
        msg="shapeless variable gelu",
    )


def bench_layernorm():
    weight = tk.random.uniform(shape=(4096,)).astype(tk.float16)
    bias = tk.random.uniform(shape=(4096,)).astype(tk.float16)
    tk.eval(weight, bias)

    def layernorm(x):
        x = x.astype(tk.float32)
        means = tk.mean(x, axis=-1, keepdims=True)
        var = tk.var(x, axis=-1, keepdims=True)
        x = (x - means) * tk.rsqrt(var + 1e-4)
        x = x.astype(tk.float16)
        return weight * x + bias

    x = tk.random.uniform(shape=(1000, 4096)).astype(tk.float16)

    def gen_fun(fun):
        def bench_fun(x):
            for _ in range(10):
                x = fun(x)
            return x

        return bench_fun

    time_fn(gen_fun(layernorm), x, msg="fixed layernorm")
    time_fn(gen_fun(tk.compile(layernorm)), x, msg="compiled fixed layernorm")

    def randint():
        return random.randint(1, x.shape[0])

    def gen_fun(fun):
        def bench_fun(x):
            x = x[: randint()]
            for _ in range(10):
                x = fun(x)
            return x

        return bench_fun

    random.seed(0)
    time_fn(gen_fun(layernorm), x, msg="variable layernorm")
    random.seed(0)
    time_fn(gen_fun(tk.compile(layernorm)), x, msg="compiled variable layernorm")
    random.seed(0)
    time_fn(
        gen_fun(tk.compile(layernorm, shapeless=True)),
        x,
        msg="shapeless variable layernorm",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compile benchmarks.")
    args = parser.parse_args()

    bench_gelu()
    bench_layernorm()
