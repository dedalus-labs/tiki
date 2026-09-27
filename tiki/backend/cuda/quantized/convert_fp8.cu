// Copyright © 2025 Apple Inc.
#include "tiki/backend/cuda/unary/unary.cuh"
#include "tiki/fast_primitives.h"

namespace tiki::core {
void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("ConvertFP8::eval_gpu");
  auto& in = inputs[0];
  auto& out = outputs[0];
  auto& s = out.primitive().stream();
  if (to_fp8_) {
    unary_op_gpu<cu::ToFP8>(inputs, out, name(), s);
  } else {
    unary_op_gpu<cu::FromFP8>(inputs, out, name(), s);
  }
}
} // namespace tiki::core
