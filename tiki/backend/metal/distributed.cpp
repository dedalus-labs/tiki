// Copyright © 2024 Apple Inc.

#include <cassert>

#include "tiki/allocator.h"
#include "tiki/backend/common/utils.h"
#include "tiki/backend/gpu/copy.h"
#include "tiki/backend/metal/device.h"
#include "tiki/backend/metal/utils.h"
#include "tiki/distributed/ops.h"
#include "tiki/distributed/primitives.h"
#include "tiki/fence.h"
#include "tiki/scheduler.h"

namespace tiki::core::distributed {

void AllReduce::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[AllReduce::eval_gpu] has no GPU implementation.");
}

void AllGather::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[AllGather::eval_gpu] has no GPU implementation.");
}

void Send::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[Send::eval_gpu] has no GPU implementation.");
}

void Recv::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[Recv::eval_gpu] has no GPU implementation.");
}

void ReduceScatter::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error(
      "[ReduceScatter::eval_gpu] has no GPU implementation.");
}

} // namespace tiki::core::distributed
