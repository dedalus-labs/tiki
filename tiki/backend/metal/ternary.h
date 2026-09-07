// Copyright © 2024 Apple Inc.

#pragma once

#include "tiki/array.h"

namespace tiki::core {

void ternary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s);

void ternary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s);

} // namespace tiki::core
