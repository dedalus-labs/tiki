// Copyright © 2026 Dedalus Labs, Inc.

#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "mlx/array.h"

namespace mlx::core::as_strided_detail {

inline int64_t add(int64_t a, int64_t b) {
  constexpr auto low = std::numeric_limits<int64_t>::min();
  constexpr auto high = std::numeric_limits<int64_t>::max();
  if ((b > 0 && a > high - b) || (b < 0 && a < low - b)) {
    throw std::overflow_error("[as_strided] Address arithmetic overflow.");
  }
  return a + b;
}

inline int64_t subtract(int64_t a, int64_t b) {
  constexpr auto low = std::numeric_limits<int64_t>::min();
  constexpr auto high = std::numeric_limits<int64_t>::max();
  if ((b < 0 && a > high + b) || (b > 0 && a < low + b)) {
    throw std::overflow_error("[as_strided] Address arithmetic overflow.");
  }
  return a - b;
}

inline int64_t scale(int64_t value, int64_t factor) {
  if (factor < 0) {
    throw std::invalid_argument("[as_strided] Scale must be nonnegative.");
  }
  if (factor != 0 &&
      (value > std::numeric_limits<int64_t>::max() / factor ||
       value < std::numeric_limits<int64_t>::min() / factor)) {
    throw std::overflow_error("[as_strided] Address arithmetic overflow.");
  }
  return value * factor;
}

struct Layout {
  int64_t lower_bytes;
  int64_t upper_bytes;
  int64_t data_size;
};

inline Layout layout(
    const Shape& shape,
    const Strides& strides,
    size_t offset,
    size_t itemsize) {
  if (shape.size() != strides.size()) {
    throw std::invalid_argument(
        "[as_strided] Shape and strides must have the same rank.");
  }
  if (std::any_of(shape.begin(), shape.end(), [](auto n) { return n < 0; })) {
    throw std::invalid_argument(
        "[as_strided] Negative dimensions not allowed.");
  }
  constexpr auto limit = std::numeric_limits<int64_t>::max();
  if (offset > limit || itemsize == 0 || itemsize > limit) {
    throw std::overflow_error(
        "[as_strided] Offset or item size is unrepresentable.");
  }
  auto bytes = static_cast<int64_t>(itemsize);
  auto origin = scale(static_cast<int64_t>(offset), bytes);
  // An empty view reads nothing, so its other extents must not decide whether it overflows.
  if (std::any_of(shape.begin(), shape.end(), [](auto n) { return n == 0; })) {
    return {origin, origin, 0};
  }
  int64_t elements = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    elements = scale(elements, shape[shape.size() - 1 - i]);
  }
  if (static_cast<uint64_t>(elements) >
      std::numeric_limits<size_t>::max() / itemsize) {
    throw std::overflow_error(
        "[as_strided] Array byte count is unrepresentable.");
  }
  int64_t low = 0, high = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    auto delta = scale(strides[i], shape[i] - 1);
    low = add(low, std::min<int64_t>(delta, 0));
    high = add(high, std::max<int64_t>(delta, 0));
  }
  return {
      add(origin, scale(low, bytes)),
      add(add(origin, scale(high, bytes)), bytes),
      add(subtract(high, low), 1)};
}

// A view must lie within the bytes of the array it was taken from. Allocator capacity is wider:
// recycled buffers keep another array's bytes past the requested size, and a gradient can only
// scatter into the input's own elements.
inline void check_extent(const Layout& layout, size_t input_bytes) {
  if (layout.lower_bytes < 0 || layout.upper_bytes < layout.lower_bytes ||
      static_cast<uint64_t>(layout.upper_bytes) > input_bytes) {
    throw std::invalid_argument(
        "[as_strided] View reaches outside its input array.");
  }
}

} // namespace mlx::core::as_strided_detail
