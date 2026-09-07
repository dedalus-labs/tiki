// Copyright © 2025 Apple Inc.

#include "tiki/backend/cuda/cuda.h"
#include "tiki/fast.h"

namespace tiki::core {

namespace cu {

bool is_available() {
  return false;
}

} // namespace cu

namespace fast {

CustomKernelFunction cuda_kernel(
    const std::string&,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const std::string&,
    const std::string&,
    bool,
    int) {
  throw std::runtime_error("[cuda_kernel] No CUDA back-end.");
}

std::vector<array> precompiled_cuda_kernel(
    const std::string&,
    const std::string&,
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    const std::vector<ScalarArg>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    int shared_memory,
    std::optional<float> init_value,
    bool ensure_row_contiguous,
    StreamOrDevice) {
  throw std::runtime_error("[cuda_kernel] No CUDA back-end.");
}

} // namespace fast

} // namespace tiki::core
