// Copyright © 2024 Apple Inc.
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "tiki/api.h"
#include "tiki/array.h"
#include "tiki/utils.h"

namespace tiki::core {

TIKI_API std::pair<std::vector<std::vector<int>>, std::string> einsum_path(
    const std::string& subscripts,
    const std::vector<array>& operands);

TIKI_API array einsum(
    const std::string& subscripts,
    const std::vector<array>& operands,
    StreamOrDevice s = {});

} // namespace tiki::core
