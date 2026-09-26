# Copyright © 2023 Apple Inc.

import argparse

import tiki as tk
from time_utils import time_fn


def time_add():
    a = tk.random.uniform(shape=(32, 1024, 1024))
    b = tk.random.uniform(shape=(32, 1024, 1024))
    tk.eval(a, b)
    time_fn(tk.add, a, b)

    aT = tk.transpose(a, [0, 2, 1])
    tk.eval(aT)

    def transpose_add(a, b):
        return tk.add(a, b)

    time_fn(transpose_add, aT, b)

    b = tk.random.uniform(shape=(1024,))
    tk.eval(b)

    def slice_add(a, b):
        return tk.add(a, b)

    time_fn(slice_add, a, b)

    b = tk.reshape(b, (1, 1024, 1))
    tk.eval(b)

    def mid_slice_add(a, b):
        return tk.add(a, b)

    time_fn(mid_slice_add, a, b)


def time_matmul():
    a = tk.random.uniform(shape=(1024, 1024))
    b = tk.random.uniform(shape=(1024, 1024))
    tk.eval(a, b)
    time_fn(tk.matmul, a, b)


def time_maximum():
    a = tk.random.uniform(shape=(32, 1024, 1024))
    b = tk.random.uniform(shape=(32, 1024, 1024))
    tk.eval(a, b)
    time_fn(tk.maximum, a, b)


def time_max():
    a = tk.random.uniform(shape=(32, 1024, 1024))
    a[1, 1] = tk.nan
    tk.eval(a)
    time_fn(tk.max, a, 0)


def time_min():
    a = tk.random.uniform(shape=(32, 1024, 1024))
    a[1, 1] = tk.nan
    tk.eval(a)
    time_fn(tk.min, a, 0)


def time_negative():
    a = tk.random.uniform(shape=(10000, 1000))
    tk.eval(a)

    def negative(a):
        return -a

    tk.eval(a)

    time_fn(negative, a)


def time_exp():
    a = tk.random.uniform(shape=(1000, 100))
    tk.eval(a)
    time_fn(tk.exp, a)


def time_logsumexp():
    a = tk.random.uniform(shape=(64, 10, 10000))
    tk.eval(a)
    time_fn(tk.logsumexp, a, axis=-1)


def time_take():
    a = tk.random.uniform(shape=(10000, 500))
    ids = tk.random.randint(low=0, high=10000, shape=(20, 10))
    ids = [tk.reshape(idx, (-1,)) for idx in ids]
    tk.eval(ids)

    def random_take():
        return [tk.take(a, idx, 0) for idx in ids]

    time_fn(random_take)


def time_reshape_transposed():
    x = tk.random.uniform(shape=(256, 256, 128))
    tk.eval(x)

    def reshape_transposed():
        return tk.reshape(tk.transpose(x, (1, 0, 2)), (-1,))

    time_fn(reshape_transposed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tiki benchmarks.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if args.gpu:
        tk.set_default_device(tk.gpu)
    else:
        tk.set_default_device(tk.cpu)

    time_add()
    time_matmul()
    time_min()
    time_max()
    time_maximum()
    time_exp()
    time_negative()
    time_logsumexp()
    time_take()
    time_reshape_transposed()
