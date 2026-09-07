// Copyright © 2024 Apple Inc.

#pragma once

#include "tiki/array.h"

namespace tiki::core {

void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s);

void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s);

} // namespace tiki::core
