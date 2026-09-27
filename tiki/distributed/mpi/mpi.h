// Copyright © 2024 Apple Inc.

#include "tiki/distributed/distributed.h"

namespace tiki::core::distributed::mpi {

using GroupImpl = tiki::core::distributed::detail::GroupImpl;

bool is_available();
std::shared_ptr<GroupImpl> init(bool strict = false);

} // namespace tiki::core::distributed::mpi
