// Copyright © 2025 Apple Inc.

#include "tiki/distributed/jaccl/jaccl.h"

namespace tiki::core::distributed::jaccl {

using GroupImpl = tiki::core::distributed::detail::GroupImpl;

bool is_available() {
  return false;
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  if (strict) {
    throw std::runtime_error("Cannot initialize jaccl distributed backend.");
  }
  return nullptr;
}

std::shared_ptr<GroupImpl> init(bool strict, AllGatherFactory /* factory */) {
  if (strict) {
    throw std::runtime_error("Cannot initialize jaccl distributed backend.");
  }
  return nullptr;
}

} // namespace tiki::core::distributed::jaccl
