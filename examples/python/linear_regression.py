# Copyright © 2023 Apple Inc.

import time

import tiki as tk

num_features = 100
num_examples = 1_000
num_iters = 10_000
lr = 0.01

# True parameters
w_star = tk.random.normal((num_features,))

# Input examples (design matrix)
X = tk.random.normal((num_examples, num_features))

# Noisy labels
eps = 1e-2 * tk.random.normal((num_examples,))
y = X @ w_star + eps

# Initialize random parameters
w = 1e-2 * tk.random.normal((num_features,))


def loss_fn(w):
    return 0.5 * tk.mean(tk.square(X @ w - y))


grad_fn = tk.grad(loss_fn)

tic = time.perf_counter()
for _ in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad
    tk.eval(w)
toc = time.perf_counter()

loss = loss_fn(w)
error_norm = tk.sum(tk.square(w - w_star)).item() ** 0.5
throughput = num_iters / (toc - tic)

print(
    f"Loss {loss.item():.5f}, L2 distance: |w-w*| = {error_norm:.5f}, "
    f"Throughput {throughput:.5f} (it/s)"
)
