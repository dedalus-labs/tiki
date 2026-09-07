// Copyright © 2024 Apple Inc.

#include <tiki/tiki.h>
#include <iostream>

namespace tk = tiki::core;

int main() {
  int batch_size = 8;
  int input_dim = 32;

  // Make the input
  tk::random::seed(42);
  auto example_x = tk::random::uniform({batch_size, input_dim});

  // Import the function
  auto forward = tk::import_function("eval_mlp.tkfn");

  // Call the imported function
  auto out = forward({example_x})[0];

  std::cout << out << std::endl;

  return 0;
}
