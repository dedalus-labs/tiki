// Copyright © 2024 Apple Inc.

#include "tiki/distributed/nccl/nccl.h"

namespace tiki::core::distributed::nccl {

using GroupImpl = tiki::core::distributed::detail::GroupImpl;

bool is_available() {
  return false;
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  if (strict) {
    throw std::runtime_error("Cannot initialize nccl distributed backend.");
  }
  return nullptr;
}

} // namespace tiki::core::distributed::nccl
