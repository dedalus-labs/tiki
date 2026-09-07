// Copyright © 2025 Apple Inc.

#pragma once

#include <memory>

#include "tiki/distributed/distributed.h"

namespace tiki::core::distributed::jaccl {

using GroupImpl = tiki::core::distributed::detail::GroupImpl;

bool is_available();
std::shared_ptr<GroupImpl> init(bool strict = false);
std::shared_ptr<GroupImpl> init(bool strict, AllGatherFactory factory);

} // namespace tiki::core::distributed::jaccl
