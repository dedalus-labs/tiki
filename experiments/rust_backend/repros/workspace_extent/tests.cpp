// Copyright (c) 2026 Dedalus Labs, Inc. All rights reserved.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cuda {
template <typename T, typename U>
auto ceil_div(T value, U divisor) {
  return value / divisor + (value % divisor != 0);
}
} // namespace cuda

using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0;
int cudaStreamIsCapturing(void*, int* status) {
  *status = cudaStreamCaptureStatusNone;
  return 0;
}
#define CHECK_CUDA_ERROR(call) assert((call) == 0)

namespace mlx::core {
using ShapeElem = int32_t;
using Shape = std::vector<ShapeElem>;
constexpr int int8 = 0;
struct Buffer {};
size_t allocated = 0;
size_t described = 0;
size_t allocations = 0;
int retained = 0;
int exposed = 0;
struct array {
  array(Buffer, const Shape& shape, int) {
    described = 1;
    for (auto extent : shape) {
      if (extent <= 0) {
        throw std::invalid_argument("nonpositive workspace dimension");
      }
      described *= static_cast<size_t>(extent);
    }
  }
};
template <typename T>
T* gpu_ptr(array&) {
  ++exposed;
  return reinterpret_cast<T*>(uintptr_t{256});
}
namespace cu {
struct CommandEncoder {
  void* stream() {
    return nullptr;
  }
  void add_temporary(const array&) {
    ++retained;
  }
  template <typename T>
  T* bind(array& value) {
    return gpu_ptr<T>(value);
  }
};
Buffer malloc_async(size_t size, CommandEncoder&) {
  allocated = size;
  ++allocations;
  return {};
}
} // namespace cu
#include "workspace.inc"
} // namespace mlx::core

int main() {
  using namespace mlx::core;
  const size_t max_shape_bytes =
      size_t{std::numeric_limits<ShapeElem>::max()} * 256;
  const std::vector<size_t> sizes{
      0,
      1,
      255,
      256,
      257,
      (size_t{1} << 31) - 256,
      (size_t{1} << 31) - 1,
      size_t{1} << 31,
      size_t{1} << 32,
      (size_t{1} << 32) + 256,
      max_shape_bytes - 1,
      max_shape_bytes,
      max_shape_bytes + 1,
      std::numeric_limits<size_t>::max() - 255,
      std::numeric_limits<size_t>::max()};
  int failed = 0;
  for (auto requested : sizes) {
    allocated = described = allocations = 0;
    retained = exposed = 0;
    cu::CommandEncoder encoder;
    void* pointer = nullptr;
    bool rejected = false;
    try {
      pointer = allocate_workspace(encoder, requested);
    } catch (const std::exception&) {
      rejected = true;
    }
    bool correct;
    if (requested > max_shape_bytes) {
      correct = rejected && allocations == 0 && retained == 0 && exposed == 0;
    } else if (requested == 0) {
      correct = !rejected && !pointer && allocations == 0 && retained == 0 &&
          exposed == 0;
    } else {
      const size_t expected = (requested / 256 + (requested % 256 != 0)) * 256;
      correct = !rejected && pointer && allocations == 1 &&
          allocated == expected && described == allocated && retained == 1 &&
          exposed == 1;
    }
    std::cout << (correct ? "  ok  " : "FAIL  ") << "workspace extent "
              << requested;
    if (!correct) {
      ++failed;
      std::cout << " (allocated=" << allocated << ", shape=" << described
                << ", rejected=" << rejected << ')';
    }
    std::cout << '\n';
  }
  return failed != 0;
}
