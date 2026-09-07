# Copyright © 2024 Apple Inc.

import time

import tiki as tk
import numpy as np


def timeit(fn, its=100, args=[]):
    for _ in range(5):
        fn(*args)
    tic = time.perf_counter()
    for _ in range(its):
        fn(*args)
    toc = time.perf_counter()
    return 1e3 * (toc - tic) / its


def time_little_einsum_path():
    subscripts = "ik,kj->ij"
    x = tk.ones((32, 32))
    y = tk.ones((32, 32))
    mx_time = timeit(tk.einsum_path, args=(subscripts, x, y))

    x = np.array(x)
    y = np.array(y)
    np_time = timeit(np.einsum_path, args=(subscripts, x, y))
    print("Timing little einsum path...")
    print(f"Tiki ... {mx_time:.3f} ms")
    print(f"NumPy... {np_time:.3f} ms")


def time_big_einsum_path():
    chars = list("abcdefgh")
    char_to_dim = {c: v for v, c in enumerate(chars)}

    num_inputs = 10
    inputs = []
    subscripts = []
    for _ in range(num_inputs):
        subscript = np.random.choice(chars, size=5, replace=False).tolist()
        subscripts.append("".join(subscript))
        inputs.append(np.ones(list(char_to_dim[c] for c in subscript)))
    subscripts = ",".join(subscripts)

    np_time = timeit(np.einsum_path, args=(subscripts, *inputs))

    inputs = [tk.array(x) for x in inputs]
    mx_time = timeit(tk.einsum_path, args=(subscripts, *inputs))
    print("Timing big einsum path...")
    print(f"Tiki ... {mx_time:.3f} ms")
    print(f"NumPy... {np_time:.3f} ms")


def time_attention():
    def regular_attention(x):
        # shape [batch, sequence, num_heads, head_dim]
        queries, keys, values = x, x, x
        scores = queries.transpose(0, 2, 1, 3) @ keys.transpose(0, 2, 3, 1)
        scores = tk.softmax(scores, axis=-1)
        output = (scores @ values.transpose(0, 2, 1, 3)).swapaxes(1, 2)
        tk.eval(output)

    def einsum_attention(x):
        # shape [batch, sequence, num_heads, head_dim]
        queries, keys, values = x, x, x
        scores = tk.einsum("itjk,iujk->ijtu", queries, keys)
        scores = tk.softmax(scores, axis=-1)
        output = tk.einsum("ijtu,iujk->itjk", scores, values)
        tk.eval(output)

    x = tk.random.uniform(shape=(8, 512, 32, 128))

    regular_time = timeit(regular_attention, args=(x,))
    ein_time = timeit(einsum_attention, args=(x,))
    print("Timing einsum attention...")
    print(f"Regular ... {regular_time:.3f} ms")
    print(f"Einsum ... {ein_time:.3f} ms")


if __name__ == "__main__":
    time_little_einsum_path()
    time_big_einsum_path()
    time_attention()
