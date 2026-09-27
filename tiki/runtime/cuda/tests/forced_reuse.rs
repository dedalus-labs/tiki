// Copyright © 2026 Dedalus Labs, Inc.

//! GPU proofs for the migration contract. These need a CUDA device, so they
//! run on the GH200 and stay ignored elsewhere:
//! `cargo test --release -- --ignored`.

use std::sync::Mutex;

use tiki_cuda_runtime as rt;

#[path = "support/device.rs"]
pub(crate) mod device;

const BYTES: usize = 1 << 20;
const ROUNDS: usize = 100;

// Each test owns the process-wide cache until its assertions complete.
pub(crate) static TEST_SESSION: Mutex<()> = Mutex::new(());

fn fill(address: usize, value: u8, stream: usize) {
    device::fill(address, value, BYTES, stream);
}

fn holds(ptr: usize, pattern: u8) -> bool {
    // SAFETY: `ptr` is a live unified allocation of BYTES bytes that the test keeps alive.
    unsafe { std::slice::from_raw_parts(ptr as *const u8, BYTES) }.iter().all(|&x| x == pattern)
}

// Invariant: blocking export keeps bytes after the device address is reused.
// Witness: 100 synchronized fill/export/overwrite rounds preserve each pattern.
#[test]
#[ignore = "needs a CUDA device"]
pub(crate) fn blocking_export_survives_address_reuse() {
    let _session = TEST_SESSION.lock().unwrap();
    rt::init().expect("init");
    let stream = device::stream();
    let mut reused = 0;
    for round in 0..ROUNDS {
        let pattern = (round % 251) as u8;
        let a = rt::allocator().allocate(BYTES, 0, stream).expect("allocate");
        let device_ptr = a.data_ptr();
        assert_eq!(a.device(), 0, "round {round}");
        fill(device_ptr, pattern, stream);
        device::sync(stream);
        let host = a.host_ptr().expect("host_ptr");
        assert_eq!(a.device(), -1);
        let b = rt::allocator().allocate(BYTES, 0, stream).expect("allocate");
        reused += usize::from(b.data_ptr() == device_ptr);
        fill(b.data_ptr(), pattern.wrapping_add(1), stream);
        device::sync(stream);
        assert!(holds(host, pattern), "round {round}");
        rt::allocator().release(b).expect("release");
        rt::allocator().release(a).expect("release");
        rt::allocator().clear_cache().expect("clear");
    }
    assert!(reused > 0, "the allocator never reused the exported address");
}

// Invariant: stream completion makes exported bytes visible before source reuse.
// Witness: fill, migrate, reallocate, overwrite, and sync preserve the first fill.
#[test]
#[ignore = "needs a CUDA device"]
pub(crate) fn stream_ordered_export_keeps_data() {
    let _session = TEST_SESSION.lock().unwrap();
    rt::init().expect("init");
    let stream = device::stream();
    for round in 0..ROUNDS {
        let pattern = (round % 251) as u8;
        let a = rt::allocator().allocate(BYTES, 0, stream).expect("allocate");
        assert_eq!(a.device(), 0, "round {round}");
        fill(a.data_ptr(), pattern, stream);
        a.migrate_on(stream).expect("migrate_on");
        let b = rt::allocator().allocate(BYTES, 0, stream).expect("allocate");
        fill(b.data_ptr(), pattern.wrapping_add(1), stream);
        device::sync(stream);
        assert!(holds(a.data_ptr(), pattern), "round {round}");
        rt::allocator().release(b).expect("release");
        rt::allocator().release(a).expect("release");
        rt::allocator().clear_cache().expect("clear");
    }
}

// Invariant: release retains reusable capacity until the cache is cleared.
// Witness: 100 bytes round to 128 and a 90-byte request reuses that address.
#[test]
#[ignore = "needs a CUDA device"]
pub(crate) fn cache_accounting() {
    let _session = TEST_SESSION.lock().unwrap();
    rt::init().expect("init");
    let runtime = rt::allocator();
    runtime.clear_cache().expect("clear");
    let before = runtime.cache_memory();
    let a = runtime.allocate(100, -1, 0).expect("allocate");
    assert_eq!(a.size(), 128);
    assert_eq!(a.device(), -1);
    let ptr = a.data_ptr();
    runtime.release(a).expect("release");
    assert_eq!(runtime.cache_memory(), before + 128);
    let again = runtime.allocate(90, -1, 0).expect("allocate");
    assert_eq!(again.data_ptr(), ptr);
    assert_eq!(runtime.cache_memory(), before);
    runtime.release(again).expect("release");
    runtime.clear_cache().expect("clear");
    assert_eq!(runtime.cache_memory(), 0);
}
