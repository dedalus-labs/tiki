// Copyright © 2025 Apple Inc.
#pragma once

#include "tiki/backend/cuda/device.h"
#include "tiki/primitives.h"

#include <optional>

namespace tiki::core {

struct GemmScalars {
  std::optional<array> alpha_device;
  std::optional<array> beta_device;

  bool has_values() const {
    return alpha_device.has_value();
  }
};

void qqmm_impl(
    cu::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale,
    QuantizationMode mode,
    const GemmScalars& scalars = {});

} // namespace tiki::core
