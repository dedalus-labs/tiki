// Copyright © 2026 Dedalus Labs, Inc.

//! Host contract tests using the production allocator and a simulated CUDA driver.
//! Run without CUDA: rustc --edition=2024 --test tests/host.rs -o /tmp/cuda-host
//! Then run /tmp/cuda-host.

// `support/device.rs` declares the few driver calls the GPU tests stage with
// by hand; here they share one crate with the generated declarations.
#![allow(dead_code, clashing_extern_declarations)]

extern crate self as tiki_cuda_runtime;

#[path = "../src/allocation.rs"]
mod allocation;
#[path = "../src/allocator.rs"]
mod allocator;
#[path = "../src/batch.rs"]
mod batch;
#[path = "../src/cache.rs"]
mod cache;
#[path = "../src/completion.rs"]
mod completion;
#[path = "support/cuda.rs"]
mod cuda;
#[path = "../../cuda-sys/src/lib.rs"]
mod driver;
#[path = "../src/event.rs"]
mod event;
#[path = "../src/pool.rs"]
mod pool;
#[path = "../src/runtime.rs"]
mod runtime;

pub use allocation::Allocation;
pub use batch::Batch;
pub use runtime::{allocator, init};

#[path = "forced_reuse.rs"]
mod forced_reuse;
use forced_reuse::device;

fn batch_of(handler: impl FnOnce() + Send + 'static) -> Batch {
    let mut batch = Batch::new();
    batch.on_complete(handler);
    batch
}

// Invariant: retained capacity drives both the handle and memory accounting.
// Witness: a two-page request reuses three pages through two cache cycles.
#[test]
fn invariant_reuse_preserves_capacity() {
    let _session = forced_reuse::TEST_SESSION.lock().unwrap();
    init().unwrap();
    let allocator = allocator::Allocator::new().unwrap();
    let large = allocator.allocate(49152, -1, 0).unwrap();
    let ptr = large.data_ptr();
    allocator.release(large).unwrap();
    for _ in 0..2 {
        let small = allocator.allocate(32768, -1, 0).unwrap();
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
    let _session = forced_reuse::TEST_SESSION.lock().unwrap();
    device::set_current(1);
    init().unwrap();
    let allocator = allocator::Allocator::new().unwrap();
    assert_eq!(cuda::current_device(), 1, "initialization");
    device::set_current(0);
    let source_stream = driver::with_device(cuda::current_device(), driver::create_stream).unwrap();
    device::set_current(1);
    let target_stream = driver::with_device(cuda::current_device(), driver::create_stream).unwrap();
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
    let _session = forced_reuse::TEST_SESSION.lock().unwrap();
    device::set_current(1);
    cuda::fail_next("cuStreamCreate");
    assert!(allocator::Allocator::new().is_err());
    assert_eq!(cuda::current_device(), 1, "failed initialization");
    init().unwrap();
    let allocator = allocator::Allocator::new().unwrap();
    device::set_current(0);
    let stream = driver::with_device(cuda::current_device(), driver::create_stream).unwrap();
    device::set_current(1);
    cuda::fail_next("cuMemAllocAsync");
    assert!(allocator.allocate(32768, 0, stream).is_err());
    assert_eq!(cuda::current_device(), 1, "failed allocation");
    let allocation = allocator.allocate(32768, 0, stream).unwrap();
    cuda::fail_next("cuMemcpyAsync");
    assert!(allocator.migrate(&allocation, None).is_err());
    assert_eq!(cuda::current_device(), 1, "failed migration");
    allocator.release(allocation).unwrap();
    allocator.clear_cache().unwrap();
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

#[test]
fn invariant_event_calls_preserve_current_device() {
    device::set_current(1);
    let event = event::Event::take(0).unwrap();
    assert_eq!(cuda::current_device(), 1, "event creation");
    event.record(1usize).unwrap();
    assert_eq!(cuda::current_device(), 1, "event recording");
    cuda::fail_next("cuEventRecord");
    assert!(event.record(1usize).is_err());
    assert_eq!(cuda::current_device(), 1, "failed event recording");
}

#[test]
fn invariant_completion_preserves_current_device() {
    let _session = forced_reuse::TEST_SESSION.lock().unwrap();
    init().unwrap();
    device::set_current(1);
    let stream = 1usize;
    let (send, receive) = std::sync::mpsc::channel();
    let batch = batch_of(move || send.send(()).unwrap());
    runtime::completion().commit(0, stream, batch).unwrap();
    assert_eq!(cuda::current_device(), 1, "commit");
    runtime::completion().order_after_latest(stream, 2usize).unwrap();
    assert_eq!(cuda::current_device(), 1, "ordering");
    receive.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
    runtime::completion().wait_stream_idle(stream);
}

#[test]
fn invariant_failed_enqueue_retains_batch() {
    let _session = forced_reuse::TEST_SESSION.lock().unwrap();
    init().unwrap();
    for call in ["cuEventRecord", "cuStreamCreate", "cuStreamWaitEvent", "cuLaunchHostFunc"] {
        // Use an unpublished completion owner to exercise each fresh signal stream.
        let completion = completion::Completion::new();
        device::set_current(1);
        let resource = std::sync::Arc::new(());
        let retained = resource.clone();
        let batch = batch_of(move || drop(retained));
        cuda::fail_next(call);
        let error = completion.commit(0, 1usize, batch).unwrap_err();
        assert_eq!(error.call, call);
        assert_eq!(cuda::current_device(), 1, "{call}");
        assert_eq!(std::sync::Arc::strong_count(&resource), 2, "{call}");
        completion.wait_stream_idle(1usize);
    }
}

#[test]
fn invariant_enqueued_callback_owns_retirement_after_restore_error() {
    let _session = forced_reuse::TEST_SESSION.lock().unwrap();
    init().unwrap();
    device::set_current(1);
    let (send, receive) = std::sync::mpsc::channel();
    let batch = batch_of(move || send.send(()).unwrap());
    cuda::fail_restore_after("cuLaunchHostFunc");
    let error = runtime::completion().commit(0, 1usize, batch).unwrap_err();
    assert_eq!(error.call, "cuCtxPopCurrent");
    receive.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
    runtime::completion().wait_stream_idle(1usize);
}

#[test]
fn invariant_reaper_selects_each_batch_device() {
    let _session = forced_reuse::TEST_SESSION.lock().unwrap();
    init().unwrap();
    for device in [1, 0, 1] {
        let (send, receive) = std::sync::mpsc::channel();
        let batch = batch_of(move || {
            send.send(cuda::current_device()).unwrap();
            device::set_current(7);
        });
        let stream = device as usize + 1;
        runtime::completion().commit(device, stream, batch).unwrap();
        assert_eq!(receive.recv_timeout(std::time::Duration::from_secs(2)).unwrap(), device);
        runtime::completion().wait_stream_idle(stream);
    }
}

#[test]
fn invariant_reused_capacity_follows_new_stream_ordering() {
    let _session = forced_reuse::TEST_SESSION.lock().unwrap();
    init().unwrap();
    device::set_current(0);
    let first = driver::with_device(cuda::current_device(), driver::create_stream).unwrap();
    let second = driver::with_device(cuda::current_device(), driver::create_stream).unwrap();
    let allocator = allocator::Allocator::new().unwrap();
    let allocation = allocator.allocate(49152, 0, first).unwrap();
    let ptr = allocation.data_ptr();
    allocator.release(allocation).unwrap();
    let reused = allocator.allocate(32768, 0, second).unwrap();
    assert_eq!(reused.data_ptr(), ptr);
    assert_eq!(reused.size(), 49152);
    assert_eq!(allocator.active_memory(), 49152);
    runtime::completion().commit(0, second, Batch::new()).unwrap();
    let before = cuda::event_waits();
    allocator.migrate(&reused, None).unwrap();
    assert_eq!(cuda::event_waits(), before + 1);
    allocator.release(reused).unwrap();
    allocator.clear_cache().unwrap();
}

#[test]
fn invariant_reaper_device_failure_stops_the_process() {
    const CHILD: &str = "TIKI_HOST_REAPER_FAILURE";
    if std::env::var_os(CHILD).is_some() {
        init().unwrap();
        let batch = batch_of(|| cuda::fail_next("cuCtxPopCurrent"));
        runtime::completion().commit(0, 1usize, batch).unwrap();
        // A worker panic must fail this test instead of leaving the child blocked.
        std::thread::sleep(std::time::Duration::from_secs(2));
        panic!("worker failure did not stop the process");
    }
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "invariant_reaper_device_failure_stops_the_process", "--nocapture"])
        .env(CHILD, "1")
        .output()
        .unwrap();
    assert!(!child.status.success());
    assert!(
        String::from_utf8(child.stderr)
            .unwrap()
            .contains("completion worker: cuCtxPopCurrent failed:")
    );
}
