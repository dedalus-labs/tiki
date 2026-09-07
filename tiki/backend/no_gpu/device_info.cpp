// Copyright © 2026 Apple Inc.

#include "tiki/backend/gpu/device_info.h"

namespace tiki::core::gpu {

bool is_available() {
  return false;
}

int device_count() {
  return 0;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int /* device_index */) {
  static std::unordered_map<std::string, std::variant<std::string, size_t>>
      empty;
  return empty;
}

} // namespace tiki::core::gpu
