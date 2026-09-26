// Copyright © 2026 Apple Inc.

#pragma once

#include "tiki/array.h"
#include "tiki/backend/cuda/device.h"

namespace tiki::core {

void apply_block_mask(
    cu::CommandEncoder& encoder,
    array& data,
    const array& mask,
    int block_size,
    int64_t rows,
    int64_t cols,
    int64_t batch_count);

array copy_with_block_mask(
    cu::CommandEncoder& encoder,
    const array& src,
    const array& mask,
    int block_size,
    int64_t rows,
    int64_t cols,
    int64_t batch_count);

} // namespace tiki::core
