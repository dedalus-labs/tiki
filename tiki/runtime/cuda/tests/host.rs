// Copyright © 2026 Dedalus Labs, Inc.

//! Host contract tests: the production runtime and driver adapter against a
//! simulated CUDA driver. Run without CUDA:
//! rustc --edition=2024 --test tests/host.rs -o /tmp/cuda-host
//! Then run /tmp/cuda-host.

#![allow(dead_code, unused_imports)]

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
#[path = "../src/runtime.rs"]
mod runtime;

use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use batch::Batch;
use driver::{CopyError, Device, DeviceMemory, Event, Stream};

/// The simulated driver's counters and the runtime are process-wide, so each
/// test that reads them owns them until its assertions complete.
static SESSION: Mutex<()> = Mutex::new(());

const RETIREMENT: Duration = Duration::from_secs(2);

fn stream_on(ordinal: i32) -> Arc<Stream> {
    Arc::new(Stream::new(Device::open(ordinal).unwrap()).unwrap())
}

fn batch_of(handler: impl FnOnce() + Send + 'static) -> Batch {
    let mut batch = Batch::new();
    batch.on_complete(handler);
    batch
}

/// Run this test binary's `test` in a child process with `marker` set, and
/// return the child's standard error after checking it failed.
fn failing_child(test: &str, marker: &str) -> String {
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(marker, "1")
        .output()
        .unwrap();
    assert!(!child.status.success(), "the child process kept running");
    String::from_utf8(child.stderr).unwrap()
}

// Invariant: reused memory keeps its size class in the handle and in the accounting.
// Witness: a two-page request reuses three pages through two cache cycles
// without a second driver allocation.
#[test]
fn invariant_reuse_preserves_capacity() {
    let _session = SESSION.lock().unwrap();
    let stream = stream_on(0);
    let allocator = allocator::Allocator::new().unwrap();
    let large = allocator.allocate(49152, &stream).unwrap();
    allocator.release(large);
    let (allocations, ..) = cuda::live();
    for _ in 0..2 {
        let small = allocator.allocate(32768, &stream).unwrap();
        assert_eq!(small.size(), 49152);
        assert_eq!(cuda::live().0, allocations, "reuse allocated again");
        assert_eq!(allocator.active_memory(), 49152);
        assert_eq!(allocator.cache_memory(), 0);
        allocator.release(small);
        assert_eq!(allocator.active_memory(), 0);
        assert_eq!(allocator.cache_memory(), 49152);
    }
    allocator.clear_cache();
    assert_eq!(cuda::live().0, allocations - 1, "clearing the cache kept memory");
}

// Invariant: every call restores the caller's current device.
// Witness: device 1 stays current across allocation, fills, copies, events,
// commits, and frees that all run on device 0.
#[test]
fn invariant_calls_preserve_current_device() {
    let _session = SESSION.lock().unwrap();
    runtime::init().unwrap();
    cuda::set_current(1);
    let stream = stream_on(0);
    assert_eq!(cuda::current_device(), 1, "stream creation");
    let allocator = allocator::Allocator::new().unwrap();
    let mut source = allocator.allocate(64, &stream).unwrap();
    let mut target = allocator.allocate(64, &stream).unwrap();
    assert_eq!(cuda::current_device(), 1, "allocation");
    stream.fill(source.memory_mut(), 7).unwrap();
    stream.copy(source.memory(), target.memory_mut()).unwrap();
    let mut host = [0u8; 64];
    stream.read(target.memory(), &mut host).unwrap();
    assert_eq!(host, [7; 64]);
    stream.write(&host, source.memory_mut()).unwrap();
    assert_eq!(cuda::current_device(), 1, "fills and copies");
    let event = Event::new(Device::open(0).unwrap()).unwrap();
    event.record(&stream).unwrap();
    assert_eq!(cuda::current_device(), 1, "events");
    let (send, receive) = mpsc::channel();
    let _ =
        runtime::completion().commit(&stream, batch_of(move || send.send(()).unwrap())).unwrap();
    receive.recv_timeout(RETIREMENT).unwrap();
    runtime::completion().wait_stream_idle(&stream);
    assert_eq!(cuda::current_device(), 1, "commit");
    allocator.release(source);
    allocator.release(target);
    allocator.clear_cache();
    drop(event);
    assert_eq!(cuda::current_device(), 1, "frees");
}

// Invariant: failed calls also restore the caller's current device.
// Witness: an allocation, a copy, and an event record fail after a device switch.
#[test]
fn invariant_errors_preserve_current_device() {
    let _session = SESSION.lock().unwrap();
    cuda::set_current(1);
    let stream = stream_on(0);
    cuda::fail_next("cuMemAllocAsync");
    assert!(DeviceMemory::allocate(&stream, 64).is_err());
    assert_eq!(cuda::current_device(), 1, "failed allocation");
    let source = DeviceMemory::allocate(&stream, 64).unwrap();
    let mut target = DeviceMemory::allocate(&stream, 64).unwrap();
    cuda::fail_next("cuMemcpyDtoDAsync");
    assert!(stream.copy(&source, &mut target).is_err());
    assert_eq!(cuda::current_device(), 1, "failed copy");
    let event = Event::new(Device::open(0).unwrap()).unwrap();
    cuda::fail_next("cuEventRecord");
    assert!(event.record(&stream).is_err());
    assert_eq!(cuda::current_device(), 1, "failed event record");
}

// Invariant: fills and copies refuse lengths and devices that do not match
// before any driver call, and a round trip returns the bytes written.
// Witness: 32 bytes into 16, a 16-element read from 8 bytes, a device-1
// stream on device-0 memory, then 16 bytes written, copied, and read back.
#[test]
fn invariant_copies_check_lengths_and_devices() {
    let _session = SESSION.lock().unwrap();
    let stream = stream_on(0);
    let mut small = DeviceMemory::allocate(&stream, 16).unwrap();
    let error = stream.write(&[0u8; 32], &mut small).unwrap_err();
    assert_eq!(error, CopyError::Length { requested: 32, available: 16 });
    let tiny = DeviceMemory::allocate(&stream, 8).unwrap();
    let error = stream.read(&tiny, &mut [0u8; 16]).unwrap_err();
    assert_eq!(error, CopyError::Length { requested: 16, available: 8 });
    let other = stream_on(1);
    let error = other.fill(&mut small, 1).unwrap_err();
    assert!(matches!(error, CopyError::Device { .. }));

    let values: Vec<u32> = (0..4).collect();
    stream.write(&values, &mut small).unwrap();
    let mut copy = DeviceMemory::allocate(&stream, 16).unwrap();
    stream.copy(&small, &mut copy).unwrap();
    let mut read = [0u32; 4];
    stream.read(&copy, &mut read).unwrap();
    assert_eq!(read.to_vec(), values);
}

// Invariant: memory reused on another stream waits for its last write and
// for its reads since, and dropping it frees it after the same uses.
// Witness: memory filled and then copied from on stream A is reused on
// stream B; B's fill waits on both events, and the drop waits on B's fill.
#[test]
fn invariant_reused_memory_waits_for_earlier_uses() {
    let _session = SESSION.lock().unwrap();
    let (first, second) = (stream_on(0), stream_on(0));
    let allocator = allocator::Allocator::new().unwrap();
    let mut shared = allocator.allocate(4096, &first).unwrap();
    let mut target = allocator.allocate(4096, &first).unwrap();
    first.fill(shared.memory_mut(), 3).unwrap();
    first.copy(shared.memory(), target.memory_mut()).unwrap();
    allocator.release(shared);
    let mut reused = allocator.allocate(4096, &second).unwrap();
    assert_eq!(allocator.cache_memory(), 0, "the released memory was not reused");
    let before = cuda::waits();
    second.fill(reused.memory_mut(), 9).unwrap();
    assert_eq!(cuda::waits() - before, 2, "the fill must wait for the write and the read");
    let (allocations, ..) = cuda::live();
    let before = cuda::waits();
    drop(reused.into_memory());
    assert_eq!(cuda::waits() - before, 1, "the free must wait for the fill");
    assert_eq!(cuda::live().0, allocations - 1);
    allocator.release(target);
    allocator.clear_cache();
}

// Invariant: a batch whose callback could not be enqueued is retained for
// the process and the commit names the failed call.
// Witness: each driver call of a commit fails once; the batch's value stays
// shared, and waiting for the stream to go idle returns.
#[test]
fn invariant_failed_enqueue_retains_batch() {
    let _session = SESSION.lock().unwrap();
    runtime::init().unwrap();
    let stream = stream_on(0);
    for call in [
        "cuEventCreate",
        "cuEventRecord",
        "cuStreamCreate",
        "cuStreamWaitEvent",
        "cuLaunchHostFunc",
    ] {
        // A fresh completion owner creates a fresh signal stream.
        let completion = completion::Completion::new();
        let resource = Arc::new(());
        let retained = resource.clone();
        cuda::fail_next(call);
        let error = completion.commit(&stream, batch_of(move || drop(retained))).unwrap_err();
        assert_eq!(error.call, call);
        assert_eq!(Arc::strong_count(&resource), 2, "{call}");
        completion.wait_stream_idle(&stream);
    }
}

// Invariant: a context that cannot be restored stops the process.
// Witness: a child process whose next context pop fails aborts inside stream
// creation and reports the failed restore.
#[test]
fn invariant_failed_restore_stops_the_process() {
    const CHILD: &str = "TIKI_HOST_RESTORE_FAILURE";
    if std::env::var_os(CHILD).is_some() {
        cuda::fail_next("cuCtxPopCurrent");
        let _ = Stream::new(Device::open(0).unwrap());
        panic!("a failed restore did not stop the process");
    }
    let stderr = failing_child("invariant_failed_restore_stops_the_process", CHILD);
    assert!(stderr.contains("cannot restore the caller's CUDA context"), "{stderr}");
}

// Invariant: a panicking batch handler stops the process instead of leaving
// every retirement waiter blocked.
// Witness: a child process commits a batch whose handler panics.
#[test]
fn invariant_panicking_handler_stops_the_process() {
    const CHILD: &str = "TIKI_HOST_HANDLER_PANIC";
    if std::env::var_os(CHILD).is_some() {
        runtime::init().unwrap();
        let stream = stream_on(0);
        let _ = runtime::completion().commit(&stream, batch_of(|| panic!("handler"))).unwrap();
        std::thread::sleep(RETIREMENT);
        panic!("a panicking handler did not stop the process");
    }
    let stderr = failing_child("invariant_panicking_handler_stops_the_process", CHILD);
    assert!(stderr.contains("completion worker: a batch handler panicked"), "{stderr}");
}
