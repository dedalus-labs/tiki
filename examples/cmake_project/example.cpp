// Copyright © 2024 Apple Inc.

#include <iostream>

#include "tiki/tiki.h"

namespace tk = tiki::core;

int main() {
  auto x = tk::array({1, 2, 3});
  auto y = tk::array({1, 2, 3});
  std::cout << x + y << std::endl;
  return 0;
}
