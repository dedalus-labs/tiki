// Copyright © 2026 Dedalus Labs, Inc.

//! GPU proofs for the completion runtime. They need CUDA device 0 and fail
//! by name without one: `cargo test --release -- --test-threads=1`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tiki_cuda_runtime::{self as rt, Allocation, Batch, Device, Stream};

/// The runtime is process-wide, so each test owns it until its assertions complete.
static SESSION: Mutex<()> = Mutex::new(());

const BUSY_BYTES: usize = 1 << 30;
const BUSY_ROUNDS: u8 = 16;

fn stream() -> Arc<Stream> {
    let device = Device::open(0).expect("these tests need CUDA device 0");
    Arc::new(Stream::new(device).expect("stream"))
}

/// Keep `stream` busy for tens of milliseconds by filling a large allocation.
fn keep_busy(stream: &Arc<Stream>) -> Allocation {
    let mut scratch = rt::allocator().allocate(BUSY_BYTES, stream).expect("allocate");
    for round in 0..BUSY_ROUNDS {
        stream.fill(scratch.memory_mut(), round).expect("fill");
    }
    scratch
}

fn counted_batch(counter: &Arc<AtomicU64>) -> Batch {
    let counter = counter.clone();
    let mut batch = Batch::new();
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
fn batch_runs_after_device_work() {
    let _session = SESSION.lock().unwrap();
    rt::init().expect("init");
    let stream = stream();
    let scratch = keep_busy(&stream);
    let counter = Arc::new(AtomicU64::new(0));
    let id = rt::completion().commit(&stream, counted_batch(&counter)).expect("commit");
    assert!(id > 0);
    assert_eq!(counter.load(Ordering::SeqCst), 0, "batch ran before the device finished");
    rt::completion().wait_stream_idle(&stream);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    rt::allocator().release(scratch);
}

// Invariant: batches complete in device order per stream, not in commit
// order across streams.
// Witness: a batch committed first on a busy stream is still pending while
// a later batch on an idle stream has already run.
#[test]
fn streams_complete_independently() {
    let _session = SESSION.lock().unwrap();
    rt::init().expect("init");
    let (busy, idle) = (stream(), stream());
    let scratch = keep_busy(&busy);
    let (first, second) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let _ = rt::completion().commit(&busy, counted_batch(&first)).expect("commit");
    let _ = rt::completion().commit(&idle, counted_batch(&second)).expect("commit");
    rt::completion().wait_stream_idle(&idle);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(first.load(Ordering::SeqCst), 0, "busy stream's batch ran early");
    rt::completion().wait_stream_idle(&busy);
    assert_eq!(first.load(Ordering::SeqCst), 1);
    rt::allocator().release(scratch);
}
