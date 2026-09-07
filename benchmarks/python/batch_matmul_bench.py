# Copyright © 2023 Apple Inc.

import argparse

import tiki as tk
from time_utils import time_fn

B = 8
T = 1024
D = 512


def time_batch_matmul():
    tk.random.seed(3)
    a = tk.random.uniform(shape=(B, T, D))
    b = tk.random.uniform(shape=(D, D))
    c = tk.random.uniform(shape=(B, T, D))
    tk.eval(a, b, c)

    time_fn(tk.matmul, a, b)

    def batch_vjp_first():
        return tk.vjp(tk.matmul, [a, b], [c])[1][0]

    time_fn(batch_vjp_first)

    def batch_vjp_second():
        return tk.vjp(tk.matmul, [a, b], [c])[1][1]

    time_fn(batch_vjp_second)


def time_unbatch_matmul():
    tk.random.seed(3)
    a = tk.random.uniform(shape=(B * T, D))
    b = tk.random.uniform(shape=(D, D))
    c = tk.random.uniform(shape=(B * T, D))
    tk.eval(a, b, c)
    time_fn(tk.matmul, a, b)

    def unbatch_vjp_first():
        return tk.matmul(c, tk.transpose(b))

    time_fn(unbatch_vjp_first)

    def unbatch_vjp_second():
        return tk.matmul(tk.transpose(a), c)

    time_fn(unbatch_vjp_second)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tiki benchmarks.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if args.gpu:
        tk.set_default_device(tk.gpu)
    else:
        tk.set_default_device(tk.cpu)

    time_batch_matmul()
    time_unbatch_matmul()
