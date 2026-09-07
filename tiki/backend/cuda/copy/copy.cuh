// Copyright © 2025 Apple Inc.

#pragma once

#include "tiki/backend/cuda/device.h"
#include "tiki/backend/cuda/device/cast_op.cuh"
#include "tiki/backend/cuda/kernel_utils.cuh"
#include "tiki/backend/gpu/copy.h"
#include "tiki/dtype_utils.h"

namespace tiki::core {

void copy_contiguous(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out);

void copy_general(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out);

void copy_general_dynamic(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    const array& dynamic_offset_in,
    const array& dynamic_offset_out);

void copy_general_input(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in);

} // namespace tiki::core
