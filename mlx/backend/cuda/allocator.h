// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"

#include <cuda_runtime.h>

namespace mlx::core::cu {

class CommandEncoder;

using allocator::Buffer;

// Address of the storage for device code.
void* storage_ptr(Buffer& buffer);

// CUDA device holding the storage, or -1 for unified memory.
int storage_device(const Buffer& buffer);

// Move device storage to unified memory on |stream| without waiting.
void migrate_on(Buffer& buffer, cudaStream_t stream);

// Adapter over the Rust runtime, which owns every allocation.
class CudaAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  Buffer malloc_async(size_t size, int device, cudaStream_t stream);
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

 private:
  CudaAllocator();
  friend CudaAllocator& allocator();
};

CudaAllocator& allocator();

Buffer malloc_async(size_t size, CommandEncoder& encoder);

} // namespace mlx::core::cu
