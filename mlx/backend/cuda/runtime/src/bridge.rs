// Copyright © 2026 Dedalus Labs, Inc.

//! The CXX bridge: opaque allocations and the operations MLX's C++ needs.
//! Streams cross as integers; C++ casts them from `cudaStream_t`.

use crate::allocator::{self, Allocation};
use crate::cudart::Stream;

#[cxx::bridge(namespace = "mlx::core::cu::rt")]
mod ffi {
    extern "Rust" {
        type Allocation;

        fn init() -> Result<()>;
        fn allocate(size: usize, device: i32, stream: usize) -> Result<Box<Allocation>>;
        fn release(allocation: Box<Allocation>) -> Result<()>;

        fn size(self: &Allocation) -> usize;
        fn device(self: &Allocation) -> i32;
        fn data_ptr(self: &Allocation) -> usize;
        fn host_ptr(self: &Allocation) -> Result<usize>;
        fn migrate_on(self: &Allocation, stream: usize) -> Result<()>;

        fn active_memory() -> usize;
        fn peak_memory() -> usize;
        fn reset_peak_memory();
        fn memory_limit() -> usize;
        fn set_memory_limit(limit: usize) -> usize;
        fn cache_memory() -> usize;
        fn set_cache_limit(limit: usize) -> usize;
        fn clear_cache() -> Result<()>;
    }
}

fn init() -> Result<(), allocator::AllocError> {
    allocator::init().map_err(Into::into)
}

fn allocate(
    size: usize,
    device: i32,
    stream: usize,
) -> Result<Box<Allocation>, allocator::AllocError> {
    allocator::runtime()
        .allocate(size, device, stream as Stream)
        .map(Box::new)
}

#[allow(clippy::boxed_local)]
fn release(allocation: Box<Allocation>) -> Result<(), crate::CudaError> {
    allocator::runtime().release(*allocation)
}

fn active_memory() -> usize {
    allocator::runtime().active_memory()
}

fn peak_memory() -> usize {
    allocator::runtime().peak_memory()
}

fn reset_peak_memory() {
    allocator::runtime().reset_peak_memory()
}

fn memory_limit() -> usize {
    allocator::runtime().memory_limit()
}

fn set_memory_limit(limit: usize) -> usize {
    allocator::runtime().set_memory_limit(limit)
}

fn cache_memory() -> usize {
    allocator::runtime().cache_memory()
}

fn set_cache_limit(limit: usize) -> usize {
    allocator::runtime().set_cache_limit(limit)
}

fn clear_cache() -> Result<(), crate::CudaError> {
    allocator::runtime().clear_cache()
}
