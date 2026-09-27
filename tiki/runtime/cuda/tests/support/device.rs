// Copyright © 2026 Dedalus Labs, Inc.

//! Driver calls the GPU tests use to stage work around the runtime: a current
//! context, streams, fills, and copies. The host tests link simulated versions.

// Each test crate uses a subset of these helpers.
#![allow(dead_code)]

use std::ffi::c_void;

unsafe extern "C" {
    fn cuInit(flags: u32) -> u32;
    fn cuDevicePrimaryCtxRetain(context: *mut *mut c_void, device: i32) -> u32;
    fn cuCtxSetCurrent(context: *mut c_void) -> u32;
    fn cuStreamCreate(stream: *mut *mut c_void, flags: u32) -> u32;
    fn cuStreamSynchronize(stream: *mut c_void) -> u32;
    fn cuMemsetD8Async(address: u64, value: u8, count: usize, stream: *mut c_void) -> u32;
    fn cuMemcpyAsync(dst: u64, src: u64, count: usize, stream: *mut c_void) -> u32;
}

const STREAM_NON_BLOCKING: u32 = 1;

/// Make device `device`'s primary context current on this thread, as the C++ core does.
pub fn set_current(device: i32) {
    let mut context = std::ptr::null_mut();
    // SAFETY: `context` outlives the calls; primary contexts live for the process.
    unsafe {
        assert_eq!(cuInit(0), 0);
        assert_eq!(cuDevicePrimaryCtxRetain(&mut context, device), 0);
        assert_eq!(cuCtxSetCurrent(context), 0);
    }
}

/// A nonblocking stream on device 0.
pub fn stream() -> usize {
    set_current(0);
    let mut stream = std::ptr::null_mut();
    // SAFETY: `stream` outlives the call.
    assert_eq!(unsafe { cuStreamCreate(&mut stream, STREAM_NON_BLOCKING) }, 0);
    stream as usize
}

pub fn sync(stream: usize) {
    // SAFETY: `stream` is a live stream.
    assert_eq!(unsafe { cuStreamSynchronize(stream as *mut c_void) }, 0);
}

/// Fill `count` bytes at `address` with `value`, in stream order.
pub fn fill(address: usize, value: u8, count: usize, stream: usize) {
    // SAFETY: `address` is a live allocation of at least `count` bytes.
    assert_eq!(unsafe { cuMemsetD8Async(address as u64, value, count, stream as *mut c_void) }, 0);
}

/// Copy `count` bytes between two live allocations, in stream order.
pub fn copy(dst: usize, src: usize, count: usize, stream: usize) {
    // SAFETY: both addresses are live allocations of at least `count` bytes.
    assert_eq!(unsafe { cuMemcpyAsync(dst as u64, src as u64, count, stream as *mut c_void) }, 0);
}
