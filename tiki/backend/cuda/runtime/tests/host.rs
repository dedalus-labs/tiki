// Copyright © 2026 Dedalus Labs, Inc.

//! Host contract tests using the production allocator and simulated CUDA calls.
//! Run without CUDA: rustc --edition=2021 --test tests/host.rs -o /tmp/cuda-host
//! Then run /tmp/cuda-host.

#![allow(dead_code)]

extern crate self as tiki_cuda_runtime;

#[path = "../src/allocation.rs"]
mod allocation;
#[path = "../src/allocator.rs"]
mod allocator;
#[path = "../src/cache.rs"]
mod cache;
#[path = "support/cuda.rs"]
mod cuda;
#[path = "../src/cudart.rs"]
mod cudart;
#[path = "../src/pool.rs"]
mod pool;

pub use allocator::{init, runtime};

#[path = "forced_reuse.rs"]
mod forced_reuse;

// Invariant: retained capacity drives both the handle and memory accounting.
// Witness: a two-page request reuses three pages through two cache cycles.
#[test]
fn invariant_reuse_preserves_capacity() {
    let allocator = allocator::Allocator::new().unwrap();
    let large = allocator.allocate(49152, -1, std::ptr::null_mut()).unwrap();
    let ptr = large.data_ptr();
    allocator.release(large).unwrap();
    for _ in 0..2 {
        let small = allocator.allocate(32768, -1, std::ptr::null_mut()).unwrap();
        assert_eq!(small.data_ptr(), ptr);
        assert_eq!(small.size(), 49152);
        assert_eq!(allocator.active_memory(), 49152);
        assert_eq!(allocator.cache_memory(), 0);
        allocator.release(small).unwrap();
        assert_eq!(allocator.active_memory(), 0);
        assert_eq!(allocator.cache_memory(), 49152);
    }
    allocator.clear_cache().unwrap();
}

// Invariant: allocator calls preserve the caller's current CUDA device.
// Witness: init, fresh allocation, cross-device reuse, and host export on GPU 1.
#[test]
fn invariant_allocator_preserves_current_device() {
    cudart::set_device(1).unwrap();
    let allocator = allocator::Allocator::new().unwrap();
    assert_eq!(cuda::current_device(), 1, "initialization");
    cudart::set_device(0).unwrap();
    let source_stream = cudart::create_stream().unwrap();
    cudart::set_device(1).unwrap();
    let target_stream = cudart::create_stream().unwrap();
    let source = allocator.allocate(32768, 0, source_stream).unwrap();
    assert_eq!(cuda::current_device(), 1, "fresh allocation");
    allocator.release(source).unwrap();
    let reused = allocator.allocate(32768, 1, target_stream).unwrap();
    assert_eq!(cuda::current_device(), 1, "cross-device reuse");
    allocator.release(reused).unwrap();
    allocator.clear_cache().unwrap();
    let source = allocator.allocate(32768, 0, source_stream).unwrap();
    allocator.migrate(&source, None).unwrap();
    assert_eq!(cuda::current_device(), 1, "host export");
    allocator.release(source).unwrap();
    allocator.clear_cache().unwrap();
}

// Invariant: errors also preserve the caller's current device.
// Witness: stream creation, allocation, and migration fail after a device switch.
#[test]
fn invariant_allocator_errors_preserve_current_device() {
    cudart::set_device(1).unwrap();
    cuda::fail_next("cudaStreamCreateWithFlags");
    assert!(allocator::Allocator::new().is_err());
    assert_eq!(cuda::current_device(), 1, "failed initialization");
    let allocator = allocator::Allocator::new().unwrap();
    cudart::set_device(0).unwrap();
    let stream = cudart::create_stream().unwrap();
    cudart::set_device(1).unwrap();
    cuda::fail_next("cudaMallocAsync");
    assert!(allocator.allocate(32768, 0, stream).is_err());
    assert_eq!(cuda::current_device(), 1, "failed allocation");
    let allocation = allocator.allocate(32768, 0, stream).unwrap();
    cuda::fail_next("cudaMemcpyAsync");
    assert!(allocator.migrate(&allocation, None).is_err());
    assert_eq!(cuda::current_device(), 1, "failed migration");
    allocator.release(allocation).unwrap();
    allocator.clear_cache().unwrap();
}

// Invariant: failed device restoration cannot lose new CUDA resources.
// Witness: initialization and allocation succeed before cudaSetDevice fails.
#[test]
fn invariant_restore_errors_release_new_resources() {
    cudart::set_device(1).unwrap();
    let before = cuda::live_resources();
    cuda::fail_restore_after("tiki_cuda_mem_advise");
    let error = allocator::Allocator::new().err().unwrap();
    assert_eq!(error.call, "cudaSetDevice");
    assert_eq!(cuda::live_resources(), before, "initialization cleanup");

    let allocator = allocator::Allocator::new().unwrap();
    cudart::set_device(0).unwrap();
    let stream = cudart::create_stream().unwrap();
    cudart::set_device(1).unwrap();
    let before = cuda::live_resources();
    cuda::fail_restore_after("cudaMallocAsync");
    let error = allocator.allocate(32768, 0, stream).err().unwrap();
    let allocator::AllocError::Cuda(error) = error else {
        panic!("expected a CUDA error");
    };
    assert_eq!(error.call, "cudaSetDevice");
    assert_eq!(cuda::live_resources(), before, "allocation cleanup");
    assert_eq!(allocator.active_memory(), 0);
}

// Invariant: tests that own the global allocator cannot change each other's cache.
// Witness: repeated accounting checks overlap both export workloads.
#[test]
fn invariant_test_sessions_are_isolated() {
    forced_reuse::cache_accounting();
    std::thread::scope(|scope| {
        scope.spawn(forced_reuse::blocking_export_survives_address_reuse);
        scope.spawn(forced_reuse::stream_ordered_export_keeps_data);
        scope.spawn(|| {
            for _ in 0..100_000 {
                forced_reuse::cache_accounting();
                std::thread::yield_now();
            }
        });
    });
}
