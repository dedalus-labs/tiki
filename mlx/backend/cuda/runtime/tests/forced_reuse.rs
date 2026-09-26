// Copyright © 2026 Dedalus Labs, Inc.

//! GPU proofs for the migration contract. These need a CUDA device, so they
//! run on the GH200 and stay ignored elsewhere:
//! `cargo test --release -- --ignored`.

use std::ffi::c_void;

use tiki_cuda_runtime as rt;

extern "C" {
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaMemsetAsync(ptr: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;
}

const BYTES: usize = 1 << 20;
const ROUNDS: usize = 100;

fn stream() -> *mut c_void {
    let mut stream = std::ptr::null_mut();
    // SAFETY: `stream` outlives the call.
    assert_eq!(unsafe { cudaStreamCreateWithFlags(&mut stream, 1) }, 0);
    stream
}

fn fill(ptr: usize, value: u8, stream: *mut c_void) {
    // SAFETY: `ptr` is a live device allocation of BYTES bytes.
    assert_eq!(
        unsafe { cudaMemsetAsync(ptr as *mut c_void, value as i32, BYTES, stream) },
        0
    );
}

fn sync(stream: *mut c_void) {
    // SAFETY: `stream` is a live stream.
    assert_eq!(unsafe { cudaStreamSynchronize(stream) }, 0);
}

fn holds(ptr: usize, pattern: u8) -> bool {
    // SAFETY: `ptr` is a live unified allocation of BYTES bytes that the test keeps alive.
    unsafe { std::slice::from_raw_parts(ptr as *const u8, BYTES) }
        .iter()
        .all(|&x| x == pattern)
}

// Invariant: a blocking host export returns the bytes the device wrote even
// when the freed device address is immediately reallocated and overwritten.
// Witness: 100 rounds of fill, export, reallocate, fill with another pattern;
// every exported buffer still holds its own pattern, and the device address
// was reused at least once so the ordering was actually exercised. The cache
// is cleared each round so every round allocates device memory afresh.
#[test]
#[ignore = "needs a CUDA device"]
fn blocking_export_survives_address_reuse() {
    rt::init().expect("init");
    let stream = stream();
    let mut reused = 0;
    for round in 0..ROUNDS {
        let pattern = (round % 251) as u8;
        let a = rt::runtime().allocate(BYTES, 0, stream).expect("allocate");
        let device_ptr = a.data_ptr();
        assert_eq!(a.device(), 0, "round {round}");
        fill(device_ptr, pattern, stream);
        sync(stream);
        let host = a.host_ptr().expect("host_ptr");
        assert_eq!(a.device(), -1);
        let b = rt::runtime().allocate(BYTES, 0, stream).expect("allocate");
        reused += usize::from(b.data_ptr() == device_ptr);
        fill(b.data_ptr(), pattern.wrapping_add(1), stream);
        sync(stream);
        assert!(holds(host, pattern), "round {round}");
        rt::runtime().release(b).expect("release");
        rt::runtime().release(a).expect("release");
        rt::runtime().clear_cache().expect("clear");
    }
    assert!(
        reused > 0,
        "the allocator never reused the exported address"
    );
}

// Invariant: a stream-ordered export makes the bytes visible on the host once
// that stream completes, and the device source is released behind the copy.
// Witness: fill, migrate on the same stream, allocate again on it, overwrite,
// synchronize; the host copy holds the original pattern.
#[test]
#[ignore = "needs a CUDA device"]
fn stream_ordered_export_keeps_data() {
    rt::init().expect("init");
    let stream = stream();
    for round in 0..ROUNDS {
        let pattern = (round % 251) as u8;
        let a = rt::runtime().allocate(BYTES, 0, stream).expect("allocate");
        assert_eq!(a.device(), 0, "round {round}");
        fill(a.data_ptr(), pattern, stream);
        a.migrate_on(stream as usize).expect("migrate_on");
        let b = rt::runtime().allocate(BYTES, 0, stream).expect("allocate");
        fill(b.data_ptr(), pattern.wrapping_add(1), stream);
        sync(stream);
        assert!(holds(a.data_ptr(), pattern), "round {round}");
        rt::runtime().release(b).expect("release");
        rt::runtime().release(a).expect("release");
        rt::runtime().clear_cache().expect("clear");
    }
}

// Invariant: released storage is cached and reused for a request of the same
// class; clearing the cache releases it.
// Witness: a 100-byte unified request rounds to 128; after release the cache
// holds 128 bytes, a second request takes it back, and clear_cache empties it.
#[test]
#[ignore = "needs a CUDA device"]
fn cache_accounting() {
    rt::init().expect("init");
    let runtime = rt::runtime();
    runtime.clear_cache().expect("clear");
    let before = runtime.cache_memory();
    let a = runtime
        .allocate(100, -1, std::ptr::null_mut())
        .expect("allocate");
    assert_eq!(a.size(), 128);
    assert_eq!(a.device(), -1);
    let ptr = a.data_ptr();
    runtime.release(a).expect("release");
    assert_eq!(runtime.cache_memory(), before + 128);
    let again = runtime
        .allocate(90, -1, std::ptr::null_mut())
        .expect("allocate");
    assert_eq!(again.data_ptr(), ptr);
    assert_eq!(runtime.cache_memory(), before);
    runtime.release(again).expect("release");
    runtime.clear_cache().expect("clear");
    assert_eq!(runtime.cache_memory(), 0);
}
