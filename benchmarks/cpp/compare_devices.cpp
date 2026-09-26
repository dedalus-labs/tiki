// Copyright © 2023 Apple Inc.

#include <iostream>
#include "tiki/tiki.h"
#include "time_utils.h"

namespace mx = tiki::core;

void time_add_op() {
  std::vector<int> sizes(1, 1);
  for (int i = 0; i < 9; ++i) {
    sizes.push_back(10 * sizes.back());
  }
  set_default_device(mx::Device::cpu);
  for (auto size : sizes) {
    auto a = mx::random::uniform({size});
    auto b = mx::random::uniform({size});
    mx::eval(a, b);
    std::cout << "Size " << size << std::endl;
    TIMEM("cpu", mx::add, a, b, mx::Device::cpu);
    TIMEM("gpu", mx::add, a, b, mx::Device::gpu);
  }
}

int main() {
  time_add_op();
}
