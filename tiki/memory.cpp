// Copyright © 2026 Apple Inc.

#include <stdexcept>
#include <unordered_set>

#include "tiki/memory.h"

namespace tiki::core {

size_t get_array_buffer_size(const std::vector<array>& arrays) {
  std::unordered_set<const void*> buffers;
  buffers.reserve(arrays.size());

  size_t total = 0;
  for (const auto& arr : arrays) {
    if (arr.status() == array::Status::unscheduled) {
      throw std::invalid_argument(
          "[get_array_buffer_size] Arrays must be evaluated before querying "
          "buffer size.");
    }

    const auto& buffer = arr.buffer();
    if (buffer.ptr() != nullptr && buffers.insert(buffer.ptr()).second) {
      total += arr.buffer_size();
    }
  }
  return total;
}

} // namespace tiki::core
