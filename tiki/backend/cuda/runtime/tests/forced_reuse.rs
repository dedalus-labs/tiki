// Copyright © 2026 Dedalus Labs, Inc.

//! GPU proofs for the migration contract. These need a CUDA device, so they
//! run on the GH200 and stay ignored elsewhere:
//! `cargo test --release -- --ignored`.

use std::ffi::c_void;
use std::sync::Mutex;

use tiki_cuda_runtime as rt;

extern "C" {
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaMemsetAsync(ptr: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;
}

const BYTES: usize = 1 << 20;
const ROUNDS: usize = 100;

// Each test owns the process-wide cache until its assertions complete.
static TEST_SESSION: Mutex<()> = Mutex::new(());

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

// Invariant: blocking export keeps bytes after the device address is reused.
// Witness: 100 synchronized fill/export/overwrite rounds preserve each pattern.
#[test]
#[ignore = "needs a CUDA device"]
pub(crate) fn blocking_export_survives_address_reuse() {
    let _session = TEST_SESSION.lock().unwrap();
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

// Invariant: stream completion makes exported bytes visible before source reuse.
// Witness: fill, migrate, reallocate, overwrite, and sync preserve the first fill.
#[test]
#[ignore = "needs a CUDA device"]
pub(crate) fn stream_ordered_export_keeps_data() {
    let _session = TEST_SESSION.lock().unwrap();
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

// Invariant: release retains reusable capacity until the cache is cleared.
// Witness: 100 bytes round to 128 and a 90-byte request reuses that address.
#[test]
#[ignore = "needs a CUDA device"]
pub(crate) fn cache_accounting() {
    let _session = TEST_SESSION.lock().unwrap();
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
