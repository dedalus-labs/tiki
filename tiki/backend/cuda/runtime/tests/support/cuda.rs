// Copyright © 2026 Dedalus Labs, Inc.

//! Synchronous host memory with CUDA's thread-local current device.

#![allow(non_snake_case)]

use std::cell::Cell;
use std::ffi::{c_char, c_void};

thread_local! {
    static CURRENT: Cell<i32> = const { Cell::new(0) };
    static FAILURE: Cell<&'static str> = const { Cell::new("") };
    static RESTORE_AFTER: Cell<&'static str> = const { Cell::new("") };
    static ALLOCATIONS: Cell<i32> = const { Cell::new(0) };
    static STREAMS: Cell<i32> = const { Cell::new(0) };
}

pub fn current_device() -> i32 {
    CURRENT.get()
}

pub fn fail_next(call: &'static str) {
    FAILURE.set(call);
}

pub fn fail_restore_after(call: &'static str) {
    RESTORE_AFTER.set(call);
}

pub fn live_resources() -> (i32, i32) {
    (ALLOCATIONS.get(), STREAMS.get())
}

fn completed(call: &str) {
    if RESTORE_AFTER.get() == call {
        RESTORE_AFTER.set("");
        fail_next("cudaSetDevice");
    }
}

fn status(call: &str) -> i32 {
    if FAILURE.get() == call {
        FAILURE.set("");
        999
    } else {
        0
    }
}

extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
}

#[no_mangle]
unsafe extern "C" fn cudaGetErrorName(_: i32) -> *const c_char {
    c"cudaErrorUnknown".as_ptr()
}

#[no_mangle]
unsafe extern "C" fn cudaGetErrorString(_: i32) -> *const c_char {
    c"injected CUDA error".as_ptr()
}

#[no_mangle]
unsafe extern "C" fn cudaGetDeviceCount(count: *mut i32) -> i32 {
    *count = 2;
    0
}

#[no_mangle]
unsafe extern "C" fn cudaGetDevice(device: *mut i32) -> i32 {
    *device = CURRENT.get();
    0
}

#[no_mangle]
unsafe extern "C" fn cudaSetDevice(device: i32) -> i32 {
    let result = status("cudaSetDevice");
    if result == 0 {
        CURRENT.set(device);
    }
    result
}

#[no_mangle]
unsafe extern "C" fn cudaDeviceGetAttribute(value: *mut i32, _: i32, _: i32) -> i32 {
    *value = 1;
    0
}

#[no_mangle]
unsafe extern "C" fn cudaDeviceGetDefaultMemPool(pool: *mut *mut c_void, device: i32) -> i32 {
    *pool = (device as usize + 1) as *mut c_void;
    0
}

#[no_mangle]
unsafe extern "C" fn cudaMemPoolGetAttribute(_: *mut c_void, _: i32, value: *mut usize) -> i32 {
    *value = 0;
    0
}

#[no_mangle]
unsafe extern "C" fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32 {
    *total = 1 << 30;
    *free = *total;
    0
}

#[no_mangle]
unsafe extern "C" fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, _: u32) -> i32 {
    *stream = (CURRENT.get() as usize + 1) as *mut c_void;
    let result = status("cudaStreamCreateWithFlags");
    if result == 0 {
        STREAMS.set(STREAMS.get() + 1);
    }
    result
}

#[no_mangle]
unsafe extern "C" fn cudaStreamDestroy(_: *mut c_void) -> i32 {
    STREAMS.set(STREAMS.get() - 1);
    0
}

#[no_mangle]
unsafe extern "C" fn cudaStreamSynchronize(_: *mut c_void) -> i32 {
    status("cudaStreamSynchronize")
}

#[no_mangle]
unsafe extern "C" fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32 {
    *ptr = malloc(size);
    ALLOCATIONS.set(ALLOCATIONS.get() + 1);
    0
}

#[no_mangle]
unsafe extern "C" fn cudaMallocAsync(
    ptr: *mut *mut c_void,
    size: usize,
    stream: *mut c_void,
) -> i32 {
    let result = status("cudaMallocAsync");
    if result != 0 {
        return result;
    }
    if stream as usize != CURRENT.get() as usize + 1 {
        return 101;
    }
    let result = cudaMalloc(ptr, size);
    completed("cudaMallocAsync");
    result
}

#[no_mangle]
unsafe extern "C" fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, _: u32) -> i32 {
    cudaMalloc(ptr, size)
}

#[no_mangle]
unsafe extern "C" fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32 {
    cudaMalloc(ptr, size)
}

#[no_mangle]
unsafe extern "C" fn cudaFree(ptr: *mut c_void) -> i32 {
    free(ptr);
    ALLOCATIONS.set(ALLOCATIONS.get() - 1);
    0
}

#[no_mangle]
unsafe extern "C" fn cudaFreeAsync(ptr: *mut c_void, _: *mut c_void) -> i32 {
    cudaFree(ptr)
}

#[no_mangle]
unsafe extern "C" fn cudaFreeHost(ptr: *mut c_void) -> i32 {
    cudaFree(ptr)
}

#[no_mangle]
unsafe extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    _: i32,
    _: *mut c_void,
) -> i32 {
    let result = status("cudaMemcpyAsync");
    if result == 0 {
        std::ptr::copy_nonoverlapping(src.cast::<u8>(), dst.cast::<u8>(), size);
    }
    result
}

#[no_mangle]
unsafe extern "C" fn cudaMemsetAsync(
    ptr: *mut c_void,
    value: i32,
    size: usize,
    _: *mut c_void,
) -> i32 {
    std::ptr::write_bytes(ptr, value as u8, size);
    0
}

#[no_mangle]
unsafe extern "C" fn tiki_cuda_mem_advise(_: *const c_void, _: usize, _: i32) -> i32 {
    completed("tiki_cuda_mem_advise");
    0
}
