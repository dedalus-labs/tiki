// Copyright © 2023 Apple Inc.

#include <chrono>
#include <cmath>
#include <iostream>

#include "tiki/tiki.h"
#include "timer.h"

/**
 * An example of logistic regression with Tiki.
 */
namespace tk = tiki::core;

int main() {
  int num_features = 100;
  int num_examples = 1'000;
  int num_iters = 10'000;
  float learning_rate = 0.1;

  // True parameters
  auto w_star = tk::random::normal({num_features});

  // The input examples
  auto X = tk::random::normal({num_examples, num_features});

  // Labels
  auto y = tk::matmul(X, w_star) > 0;

  // Initialize random parameters
  tk::array w = 1e-2 * tk::random::normal({num_features});

  auto loss_fn = [&](tk::array w) {
    auto logits = tk::matmul(X, w);
    auto scale = (1.0f / num_examples);
    return scale * tk::sum(tk::logaddexp(tk::array(0.0f), logits) - y * logits);
  };

  auto grad_fn = tk::grad(loss_fn);

  auto tic = timer::time();
  for (int it = 0; it < num_iters; ++it) {
    auto grads = grad_fn(w);
    w = w - learning_rate * grads;
    tk::eval(w);
  }
  auto toc = timer::time();

  auto loss = loss_fn(w);
  auto acc = tk::sum((tk::matmul(X, w) > 0) == y) / num_examples;
  auto throughput = num_iters / timer::seconds(toc - tic);
  std::cout << "Loss " << loss << ", Accuracy, " << acc << ", Throughput "
            << throughput << " (it/s)." << std::endl;
}
