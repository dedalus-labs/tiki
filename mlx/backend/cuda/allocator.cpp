// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/memory.h"
#include "mlx/scheduler.h"

#include "tiki-cuda-runtime/src/bridge.rs.h"

namespace mlx::core {

namespace cu {

namespace {

rt::Allocation& allocation(Buffer& buffer) {
  return *static_cast<rt::Allocation*>(buffer.ptr());
}

const rt::Allocation& allocation(const Buffer& buffer) {
  return *static_cast<const rt::Allocation*>(buffer.ptr());
}

size_t handle(cudaStream_t stream) {
  return reinterpret_cast<size_t>(stream);
}

} // namespace

void* storage_ptr(Buffer& buffer) {
  return reinterpret_cast<void*>(allocation(buffer).data_ptr());
}

int storage_device(const Buffer& buffer) {
  return allocation(buffer).device();
}

void migrate_on(Buffer& buffer, cudaStream_t stream) {
  allocation(buffer).migrate_on(handle(stream));
}

CudaAllocator::CudaAllocator() {
  rt::init();
}

Buffer
CudaAllocator::malloc_async(size_t size, int device, cudaStream_t stream) {
  return Buffer{rt::allocate(size, device, handle(stream)).into_raw()};
}

Buffer CudaAllocator::malloc(size_t size) {
  return malloc_async(size, -1, nullptr);
}

void CudaAllocator::free(Buffer buffer) {
  if (auto* ptr = static_cast<rt::Allocation*>(buffer.ptr())) {
    rt::release(rust::Box<rt::Allocation>::from_raw(ptr));
  }
}

size_t CudaAllocator::size(Buffer buffer) const {
  return buffer.ptr() ? allocation(buffer).size() : 0;
}

CudaAllocator& allocator() {
  static auto* allocator_ = []() {
    // Ensure scheduler is created before allocator.
    scheduler::scheduler();
    // The allocator is leaked on exit so cached buffers need no teardown.
    return new CudaAllocator();
  }();
  return *allocator_;
}

Buffer malloc_async(size_t size, CommandEncoder& encoder) {
  return allocator().malloc_async(
      size, encoder.device().cuda_device(), encoder.stream());
}

} // namespace cu

namespace allocator {

Allocator& allocator() {
  return cu::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return reinterpret_cast<void*>(
      static_cast<cu::rt::Allocation*>(ptr_)->host_ptr());
}

bool can_reuse_alien_buffer(void* ptr) {
  return true;
}

} // namespace allocator

size_t get_active_memory() {
  cu::allocator();
  return cu::rt::active_memory();
}
size_t get_peak_memory() {
  cu::allocator();
  return cu::rt::peak_memory();
}
void reset_peak_memory() {
  cu::allocator();
  cu::rt::reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  cu::allocator();
  return cu::rt::set_memory_limit(limit);
}
size_t get_memory_limit() {
  cu::allocator();
  return cu::rt::memory_limit();
}
size_t get_cache_memory() {
  cu::allocator();
  return cu::rt::cache_memory();
}
size_t set_cache_limit(size_t limit) {
  cu::allocator();
  return cu::rt::set_cache_limit(limit);
}
void clear_cache() {
  cu::allocator();
  cu::rt::clear_cache();
}

// Not supported in CUDA.
size_t set_wired_limit(size_t) {
  return 0;
}

} // namespace mlx::core
