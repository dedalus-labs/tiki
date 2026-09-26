// Copyright © 2026 Dedalus Labs, Inc.
#include <cuda_runtime_api.h>

static int received = -1;

extern "C" {
#if CUDART_VERSION >= 13000
int cudaMemAdvise(
    const void*,
    size_t,
    cudaMemoryAdvise advice,
    cudaMemLocation location) {
  if (location.type != cudaMemLocationTypeDevice) {
    return 1;
  }
  int device = location.id;
#else
int cudaMemAdvise(const void*, size_t, cudaMemoryAdvise advice, int device) {
#endif
  if (advice != cudaMemAdviseSetAccessedBy) {
    return 1;
  }
  received = device;
  return device < 0 ? 101 : 0;
}

int received_device() {
  return received;
}

const char* cudaGetErrorName(int) {
  return "cudaErrorInvalidDevice";
}

const char* cudaGetErrorString(int) {
  return "invalid device ordinal";
}
}
