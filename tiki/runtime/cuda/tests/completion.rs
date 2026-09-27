// Copyright © 2026 Dedalus Labs, Inc.

//! GPU proofs for the completion runtime. They need a CUDA device, so they
//! run on the GH200 and stay ignored elsewhere:
//! `cargo test --release -- --ignored --test-threads=1`.

use std::sync::Mutex;

static TEST_SESSION: Mutex<()> = Mutex::new(());
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tiki_cuda_runtime as rt;

#[path = "support/device.rs"]
mod device;

const BUSY_BYTES: usize = 1 << 30;
const BUSY_ROUNDS: usize = 16;

/// Keep `stream` busy for tens of milliseconds by filling a large buffer.
fn keep_busy(stream: usize) -> rt::Allocation {
    let scratch = rt::allocator().allocate(BUSY_BYTES, 0, stream).expect("allocate");
    for round in 0..BUSY_ROUNDS {
        device::fill(scratch.data_ptr(), round as u8, BUSY_BYTES, stream);
    }
    scratch
}

fn counted_batch(counter: &Arc<AtomicU64>) -> rt::Batch {
    let counter = counter.clone();
    let mut batch = rt::Batch::new();
    batch.on_complete(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    });
    batch
}

// Invariant: a committed batch runs only after the device finishes the work
// enqueued before the commit, even though the caller keeps no handle.
// Witness: a stream busy for tens of milliseconds; the counter is still 0
// right after commit and becomes 1 once the stream is idle.
#[test]
#[ignore = "needs a CUDA device"]
fn batch_runs_after_device_work() {
    let _session = TEST_SESSION.lock().unwrap();
    rt::init().expect("init");
    let stream = device::stream();
    let scratch = keep_busy(stream);
    let counter = Arc::new(AtomicU64::new(0));
    let id = rt::completion().commit(0, stream, counted_batch(&counter)).expect("commit");
    assert!(id > 0);
    assert_eq!(counter.load(Ordering::SeqCst), 0, "batch ran before the device finished");
    rt::completion().wait_stream_idle(stream);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    device::sync(stream);
    rt::allocator().release(scratch).expect("release");
}

// Invariant: batches complete in device order per stream, not in commit
// order across streams.
// Witness: a batch committed first on a busy stream is still pending while
// a later batch on an idle stream has already run.
#[test]
#[ignore = "needs a CUDA device"]
fn streams_complete_independently() {
    let _session = TEST_SESSION.lock().unwrap();
    rt::init().expect("init");
    let (busy, idle) = (device::stream(), device::stream());
    let scratch = keep_busy(busy);
    let (first, second) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    rt::completion().commit(0, busy, counted_batch(&first)).expect("commit");
    rt::completion().commit(0, idle, counted_batch(&second)).expect("commit");
    rt::completion().wait_stream_idle(idle);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(first.load(Ordering::SeqCst), 0, "busy stream's batch ran early");
    rt::completion().wait_stream_idle(busy);
    assert_eq!(first.load(Ordering::SeqCst), 1);
    device::sync(busy);
    rt::allocator().release(scratch).expect("release");
}

// Invariant: ordering a waiter stream after a source stream's latest commit
// makes the waiter observe the source's writes.
// Witness: fill on a busy source stream, commit, order a fresh stream after
// it, copy there, and read the fill pattern back through a blocking export.
#[test]
#[ignore = "needs a CUDA device"]
fn waiter_observes_source_writes() {
    let _session = TEST_SESSION.lock().unwrap();
    rt::init().expect("init");
    let (source, waiter) = (device::stream(), device::stream());
    let scratch = keep_busy(source);
    let bytes = 1 << 20;
    let filled = rt::allocator().allocate(bytes, 0, source).expect("allocate");
    device::fill(filled.data_ptr(), 7, bytes, source);
    rt::completion().commit(0, source, rt::Batch::new()).expect("commit");
    rt::completion().order_after_latest(source, waiter).expect("order");
    let copy = rt::allocator().allocate(bytes, 0, waiter).expect("allocate");
    device::copy(copy.data_ptr(), filled.data_ptr(), bytes, waiter);
    device::sync(waiter);
    let host = copy.host_ptr().expect("host_ptr");
    // SAFETY: `host` is a live unified allocation of `bytes` bytes.
    let view = unsafe { std::slice::from_raw_parts(host as *const u8, bytes) };
    assert!(view.iter().all(|&x| x == 7));
    for allocation in [copy, filled, scratch] {
        rt::allocator().release(allocation).expect("release");
    }
}
