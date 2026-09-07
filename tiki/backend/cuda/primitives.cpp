// Copyright © 2025 Apple Inc.

#include "tiki/distributed/primitives.h"
#include <cuda_runtime.h>
#include "tiki/fast_primitives.h"
#include "tiki/primitives.h"

namespace tiki::core {

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no CUDA implementation.");    \
  }

#define NO_GPU_USE_FALLBACK(func)     \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no CUDA implementation.");   \
  }

NO_GPU_MULTI(LUF)
NO_GPU_MULTI(QRF)
NO_GPU_MULTI(SVD)
NO_GPU(Inverse)
NO_GPU_MULTI(Eig)
NO_GPU_MULTI(Eigh)

namespace distributed {
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
} // namespace distributed

} // namespace tiki::core
