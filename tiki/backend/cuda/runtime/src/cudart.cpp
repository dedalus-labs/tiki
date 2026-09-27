// Copyright © 2026 Dedalus Labs, Inc.

#include <cuda_runtime_api.h>

extern "C" int tiki_cuda_mem_advise(const void* ptr, size_t count, int device) {
  // CUDA 13 changed the argument ABI, so use the selected toolkit's header.
#if CUDART_VERSION >= 13000
  return cudaMemAdvise(
      ptr,
      count,
      cudaMemAdviseSetAccessedBy,
      cudaMemLocation{cudaMemLocationTypeDevice, device});
#else
  return cudaMemAdvise(ptr, count, cudaMemAdviseSetAccessedBy, device);
#endif
}
