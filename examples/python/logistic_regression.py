# Copyright © 2023 Apple Inc.

import time

import tiki as tk

num_features = 100
num_examples = 1_000
num_iters = 10_000
lr = 0.1

# True parameters
w_star = tk.random.normal((num_features,))

# Input examples
X = tk.random.normal((num_examples, num_features))

# Labels
y = (X @ w_star) > 0


# Initialize random parameters
w = 1e-2 * tk.random.normal((num_features,))


def loss_fn(w):
    logits = X @ w
    return tk.mean(tk.logaddexp(0.0, logits) - y * logits)


grad_fn = tk.grad(loss_fn)

tic = time.perf_counter()
for _ in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad
    tk.eval(w)

toc = time.perf_counter()

loss = loss_fn(w)
final_preds = (X @ w) > 0
acc = tk.mean(final_preds == y)

throughput = num_iters / (toc - tic)
print(
    f"Loss {loss.item():.5f}, Accuracy {acc.item():.5f} "
    f"Throughput {throughput:.5f} (it/s)"
)
