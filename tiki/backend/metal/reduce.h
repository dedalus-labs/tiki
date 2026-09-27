// Copyright @ 2023 - 2024 Apple Inc.

#pragma once

#include "tiki/backend/common/reduce.h"
#include "tiki/backend/metal/device.h"
#include "tiki/stream.h"

namespace tiki::core {

using metal::CommandEncoder;

void all_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s);

void row_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s);

void strided_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s);

} // namespace tiki::core
