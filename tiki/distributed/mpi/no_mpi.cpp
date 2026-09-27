// Copyright © 2024 Apple Inc.

#include "tiki/distributed/mpi/mpi.h"

namespace tiki::core::distributed::mpi {

using GroupImpl = tiki::core::distributed::detail::GroupImpl;

bool is_available() {
  return false;
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  if (strict) {
    throw std::runtime_error("Cannot initialize MPI");
  }
  return nullptr;
}

} // namespace tiki::core::distributed::mpi
