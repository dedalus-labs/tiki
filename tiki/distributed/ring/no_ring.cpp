// Copyright © 2024 Apple Inc.

#include "tiki/distributed/ring/ring.h"

namespace tiki::core::distributed::ring {

using GroupImpl = tiki::core::distributed::detail::GroupImpl;

bool is_available() {
  return false;
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  if (strict) {
    throw std::runtime_error("Cannot initialize ring distributed backend.");
  }
  return nullptr;
}

} // namespace tiki::core::distributed::ring
