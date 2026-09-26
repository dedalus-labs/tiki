// Copyright © 2025 Apple Inc.

#include "tiki/backend/cuda/binary/binary.cuh"

namespace tiki::core {
void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Equal::eval_gpu");
  auto& s = out.primitive().stream();
  if (equal_nan_) {
    binary_op_gpu<cu::NaNEqual>(inputs, out, name(), s);
  } else {
    binary_op_gpu<cu::Equal>(inputs, out, name(), s);
  }
}
} // namespace tiki::core
