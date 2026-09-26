// Copyright © 2024 Apple Inc.

#include <iostream>

#include "tiki/tiki.h"

namespace tk = tiki::core;

int main() {
  if (!tk::distributed::is_available()) {
    std::cout << "No communication backend found" << std::endl;
    return 1;
  }

  auto global_group = tk::distributed::init();
  std::cout << global_group.rank() << " / " << global_group.size() << std::endl;

  tk::array x = tk::ones({10});
  tk::array out = tk::distributed::all_sum(x, global_group);

  std::cout << out << std::endl;
}
