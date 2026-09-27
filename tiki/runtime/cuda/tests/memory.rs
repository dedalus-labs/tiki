// Copyright © 2026 Dedalus Labs, Inc.

//! GPU proofs that device memory orders its own uses across streams. They
//! need CUDA device 0 and fail by name without one:
//! `cargo test --release -- --test-threads=1`.

use std::sync::{Arc, Mutex};

use tiki_cuda_runtime::{self as rt, Allocation, CopyError, Device, Stream};

/// The allocator's cache is process-wide, so each test owns it until its
/// assertions complete.
static SESSION: Mutex<()> = Mutex::new(());

const BYTES: usize = 1 << 20;
const BUSY_BYTES: usize = 1 << 30;
const BUSY_ROUNDS: u8 = 16;

fn stream() -> Arc<Stream> {
    let device = Device::open(0).expect("these tests need CUDA device 0");
    Arc::new(Stream::new(device).expect("stream"))
}

/// Keep `stream` busy for tens of milliseconds, so work enqueued after this
/// is still pending when another stream acts.
fn keep_busy(stream: &Arc<Stream>) -> Allocation {
    let mut scratch = rt::allocator().allocate(BUSY_BYTES, stream).expect("allocate");
    for round in 0..BUSY_ROUNDS {
        stream.fill(scratch.memory_mut(), round).expect("fill");
    }
    scratch
}

fn read_all(stream: &Stream, allocation: &Allocation) -> Vec<u8> {
    let mut host = vec![0u8; BYTES];
    stream.read(allocation.memory(), &mut host).expect("read");
    host
}

// Invariant: a stream that copies from memory another stream is still
// writing sees the finished write, with no ordering from the caller.
// Witness: a fill of 7 queued behind tens of milliseconds of work on one
// stream, copied at once on a second stream, reads back as 7.
#[test]
fn waiter_observes_source_writes() {
    let _session = SESSION.lock().unwrap();
    rt::init().expect("init");
    let (source, waiter) = (stream(), stream());
    let scratch = keep_busy(&source);
    let mut filled = rt::allocator().allocate(BYTES, &source).expect("allocate");
    source.fill(filled.memory_mut(), 7).expect("fill");
    let mut copy = rt::allocator().allocate(BYTES, &waiter).expect("allocate");
    waiter.copy(filled.memory(), copy.memory_mut()).expect("copy");
    assert!(read_all(&waiter, &copy).iter().all(|&byte| byte == 7));
    for allocation in [copy, filled, scratch] {
        rt::allocator().release(allocation);
    }
}

// Invariant: memory released while a read of it is still pending is not
// overwritten until that read finishes, even when another stream reuses it.
// Witness: a copy out of memory holding 3 is queued behind tens of
// milliseconds of work; the memory is released, reused on a second stream,
// and filled with 9 at once; the copy still reads 3.
#[test]
fn reuse_waits_for_pending_reads() {
    let _session = SESSION.lock().unwrap();
    rt::init().expect("init");
    rt::allocator().clear_cache();
    let (first, second) = (stream(), stream());
    let mut shared = rt::allocator().allocate(BYTES, &first).expect("allocate");
    first.fill(shared.memory_mut(), 3).expect("fill");
    first.synchronize().expect("synchronize");
    let scratch = keep_busy(&first);
    let mut copy = rt::allocator().allocate(BYTES, &first).expect("allocate");
    first.copy(shared.memory(), copy.memory_mut()).expect("copy");
    rt::allocator().release(shared);
    let cached = rt::allocator().cache_memory();
    let mut reused = rt::allocator().allocate(BYTES, &second).expect("allocate");
    assert_eq!(
        rt::allocator().cache_memory(),
        cached - BYTES,
        "the released memory was not reused"
    );
    second.fill(reused.memory_mut(), 9).expect("fill");
    assert!(
        read_all(&first, &copy).iter().all(|&byte| byte == 3),
        "the reuse overwrote a pending read"
    );
    assert!(read_all(&second, &reused).iter().all(|&byte| byte == 9));
    for allocation in [reused, copy, scratch] {
        rt::allocator().release(allocation);
    }
}

// Invariant: host values written to device memory read back unchanged, and
// copies refuse lengths the memory cannot hold.
// Witness: a counting pattern of 2^18 u32 values round-trips; a write one
// element too long is refused with the requested and available bytes.
#[test]
fn host_round_trip_and_lengths() {
    let _session = SESSION.lock().unwrap();
    rt::init().expect("init");
    let stream = stream();
    let values: Vec<u32> = (0..u32::try_from(BYTES / 4).unwrap()).collect();
    let mut memory = rt::allocator().allocate(BYTES, &stream).expect("allocate");
    stream.write(&values, memory.memory_mut()).expect("write");
    let mut read = vec![0u32; values.len()];
    stream.read(memory.memory(), &mut read).expect("read");
    assert_eq!(read, values);
    let too_long = vec![0u32; values.len() + 1];
    let error = stream.write(&too_long, memory.memory_mut()).unwrap_err();
    assert_eq!(error, CopyError::Length { requested: BYTES + 4, available: BYTES });
    rt::allocator().release(memory);
}

// Invariant: active and cached bytes follow allocation and release.
// Witness: one allocation is active, then cached, then gone after clearing.
#[test]
fn cache_accounting() {
    let _session = SESSION.lock().unwrap();
    rt::init().expect("init");
    rt::allocator().clear_cache();
    let stream = stream();
    let active = rt::allocator().active_memory();
    let allocation = rt::allocator().allocate(BYTES, &stream).expect("allocate");
    assert_eq!(rt::allocator().active_memory(), active + BYTES);
    rt::allocator().release(allocation);
    assert_eq!(rt::allocator().active_memory(), active);
    assert_eq!(rt::allocator().cache_memory(), BYTES);
    rt::allocator().clear_cache();
    assert_eq!(rt::allocator().cache_memory(), 0);
}
