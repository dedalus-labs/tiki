// Copyright © 2024 Apple Inc.

#include <tiki/tiki.h>
#include <iostream>

namespace tk = tiki::core;

int main() {
  int batch_size = 8;
  int input_dim = 32;
  int output_dim = 10;

  auto state = tk::import_function("init_mlp.tkfn")({});

  // Make the input
  tk::random::seed(42);
  auto example_X = tk::random::normal({batch_size, input_dim});
  auto example_y = tk::random::randint(0, output_dim, {batch_size});

  // Import the function
  auto step = tk::import_function("train_mlp.tkfn");

  // Call the imported function
  for (int it = 0; it < 100; ++it) {
    state.insert(state.end(), {example_X, example_y});
    state = step(state);
    eval(state);
    auto loss = state.back();
    state.pop_back();
    if (it % 10 == 0) {
      std::cout << "Loss " << loss.item<float>() << std::endl;
    }
  }
  return 0;
}
