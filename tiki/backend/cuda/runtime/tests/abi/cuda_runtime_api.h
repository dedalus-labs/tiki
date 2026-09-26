// Copyright © 2026 Dedalus Labs, Inc.
#pragma once
#include <stddef.h>

// Signatures from NVIDIA runtime headers 12.9.79 and 13.0.48.
enum cudaMemoryAdvise { cudaMemAdviseSetAccessedBy = 5 };
enum cudaMemLocationType { cudaMemLocationTypeDevice = 1 };
struct cudaMemLocation {
  cudaMemLocationType type;
  int id;
};
extern "C" {
#if CUDART_VERSION >= 13000
int cudaMemAdvise(const void*, size_t, cudaMemoryAdvise, cudaMemLocation);
#else
int cudaMemAdvise(const void*, size_t, cudaMemoryAdvise, int);
#endif
}
