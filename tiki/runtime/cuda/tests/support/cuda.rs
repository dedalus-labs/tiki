// Copyright © 2026 Dedalus Labs, Inc.

//! A simulated CUDA driver: synchronous host memory, a thread-local context
//! stack, and streams and events that remember their device.

#![allow(non_snake_case)]

use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

const SUCCESS: u32 = 0;
const INVALID_DEVICE: u32 = 101;
const INJECTED: u32 = 999;

static NEXT_STREAM: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    /// Devices whose primary context is current, innermost last. The base
    /// entry is the caller's current device, as the CUDA runtime sets it.
    static STACK: RefCell<Vec<i32>> = RefCell::new(vec![0]);
    static FAILURE: Cell<&'static str> = const { Cell::new("") };
    static RESTORE_AFTER: Cell<&'static str> = const { Cell::new("") };
    static ALLOCATIONS: Cell<i32> = const { Cell::new(0) };
    static WAITS: Cell<usize> = const { Cell::new(0) };
    static STREAMS: Cell<i32> = const { Cell::new(0) };
}

/// Device of the innermost current context.
pub fn current_device() -> i32 {
    STACK.with_borrow(|stack| *stack.last().expect("a context is current"))
}

/// Make the next call named `call` fail.
pub fn fail_next(call: &'static str) {
    FAILURE.set(call);
}

/// Make the context pop after the next successful `call` fail.
pub fn fail_restore_after(call: &'static str) {
    RESTORE_AFTER.set(call);
}

pub fn live_resources() -> (i32, i32) {
    (ALLOCATIONS.get(), STREAMS.get())
}

pub fn event_waits() -> usize {
    WAITS.get()
}

fn completed(call: &str) {
    if RESTORE_AFTER.get() == call {
        RESTORE_AFTER.set("");
        fail_next("cuCtxPopCurrent");
    }
}

fn status(call: &str) -> u32 {
    if FAILURE.get() == call {
        FAILURE.set("");
        INJECTED
    } else {
        SUCCESS
    }
}

/// A stream handle encodes its device in the low byte.
fn stream_device(stream: *mut c_void) -> i32 {
    (stream as usize & 255) as i32 - 1
}

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuInit(_: u32) -> u32 {
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuGetErrorName(_: u32, text: *mut *const c_char) -> u32 {
    // SAFETY: the runtime supplies live storage for this call.
    unsafe { *text = c"CUDA_ERROR_UNKNOWN".as_ptr() };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuGetErrorString(_: u32, text: *mut *const c_char) -> u32 {
    // SAFETY: the runtime supplies live storage for this call.
    unsafe { *text = c"injected CUDA error".as_ptr() };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDeviceGetCount(count: *mut i32) -> u32 {
    // SAFETY: the runtime supplies live storage for this call.
    unsafe { *count = 8 };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDeviceGet(device: *mut i32, ordinal: i32) -> u32 {
    if !(0..8).contains(&ordinal) {
        return INVALID_DEVICE;
    }
    // SAFETY: the runtime supplies live storage for this call.
    unsafe { *device = ordinal };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDeviceGetAttribute(value: *mut i32, _: u32, _: i32) -> u32 {
    // SAFETY: the runtime supplies live storage for this call.
    unsafe { *value = 1 };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDevicePrimaryCtxRetain(context: *mut *mut c_void, device: i32) -> u32 {
    // SAFETY: the runtime supplies live storage for this call.
    unsafe { *context = (device as usize + 1) as *mut c_void };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuCtxPushCurrent_v2(context: *mut c_void) -> u32 {
    STACK.with_borrow_mut(|stack| stack.push(context as i32 - 1));
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuCtxPopCurrent_v2(context: *mut *mut c_void) -> u32 {
    let result = status("cuCtxPopCurrent");
    if result == SUCCESS {
        let device = STACK.with_borrow_mut(|stack| stack.pop().expect("a pushed context"));
        // SAFETY: the runtime supplies live storage for this call.
        unsafe { *context = (device as usize + 1) as *mut c_void };
    }
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuCtxSetCurrent(context: *mut c_void) -> u32 {
    let device = context as i32 - 1;
    STACK.with_borrow_mut(|stack| *stack.last_mut().expect("a context is current") = device);
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDeviceGetDefaultMemPool(pool: *mut *mut c_void, device: i32) -> u32 {
    // SAFETY: the runtime supplies live storage for this call.
    unsafe { *pool = (device as usize + 1) as *mut c_void };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemPoolGetAttribute(_: *mut c_void, _: u32, value: *mut u64) -> u32 {
    // SAFETY: the runtime supplies live storage for this call.
    unsafe { *value = 0 };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> u32 {
    // SAFETY: the runtime supplies live storage for this call.
    unsafe {
        *total = 1 << 30;
        *free = 1 << 30;
    }
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamCreate(stream: *mut *mut c_void, _: u32) -> u32 {
    let result = status("cuStreamCreate");
    if result == SUCCESS {
        let handle =
            (NEXT_STREAM.fetch_add(1, Ordering::Relaxed) << 8) | (current_device() as usize + 1);
        // SAFETY: the runtime supplies live storage for this call.
        unsafe { *stream = handle as *mut c_void };
        STREAMS.set(STREAMS.get() + 1);
        completed("cuStreamCreate");
    }
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamDestroy_v2(_: *mut c_void) -> u32 {
    STREAMS.set(STREAMS.get() - 1);
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamSynchronize(_: *mut c_void) -> u32 {
    status("cuStreamSynchronize")
}

fn allocate(address: *mut u64, size: usize) -> u32 {
    // SAFETY: the runtime supplies live storage for this call.
    unsafe { *address = malloc(size) as u64 };
    ALLOCATIONS.set(ALLOCATIONS.get() + 1);
    SUCCESS
}

fn release(address: u64) -> u32 {
    // SAFETY: the address came from `allocate` and is freed once.
    unsafe { free(address as *mut c_void) };
    ALLOCATIONS.set(ALLOCATIONS.get() - 1);
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemAlloc_v2(address: *mut u64, size: usize) -> u32 {
    allocate(address, size)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemAllocAsync(address: *mut u64, size: usize, stream: *mut c_void) -> u32 {
    let result = status("cuMemAllocAsync");
    if result != SUCCESS {
        return result;
    }
    if stream_device(stream) != current_device() {
        return INVALID_DEVICE;
    }
    let result = allocate(address, size);
    completed("cuMemAllocAsync");
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemAllocManaged(address: *mut u64, size: usize, _: u32) -> u32 {
    allocate(address, size)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemAllocHost_v2(address: *mut *mut c_void, size: usize) -> u32 {
    allocate(address.cast(), size)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemFree_v2(address: u64) -> u32 {
    release(address)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemFreeAsync(address: u64, _: *mut c_void) -> u32 {
    release(address)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemFreeHost(address: *mut c_void) -> u32 {
    release(address as u64)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemcpyAsync(dst: u64, src: u64, size: usize, _: *mut c_void) -> u32 {
    let result = status("cuMemcpyAsync");
    if result == SUCCESS {
        // SAFETY: both addresses are live simulated allocations of `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size) };
    }
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemsetD8Async(address: u64, value: u8, size: usize, _: *mut c_void) -> u32 {
    // SAFETY: the address is a live simulated allocation of `size` bytes.
    unsafe { std::ptr::write_bytes(address as *mut u8, value, size) };
    SUCCESS
}

#[repr(C)]
pub struct Location {
    kind: u32,
    id: i32,
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemAdvise_v2(_: u64, _: usize, _: u32, _: Location) -> u32 {
    completed("cuMemAdvise");
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventCreate(event: *mut *mut c_void, _: u32) -> u32 {
    let result = status("cuEventCreate");
    if result == SUCCESS {
        // SAFETY: the runtime supplies live storage for this call.
        unsafe { *event = (current_device() as usize + 1) as *mut c_void };
        completed("cuEventCreate");
    }
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> u32 {
    let result = status("cuEventRecord");
    if result != SUCCESS {
        return result;
    }
    let device = event as i32 - 1;
    if device != stream_device(stream) || device != current_device() {
        return INVALID_DEVICE;
    }
    completed("cuEventRecord");
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventQuery(_: *mut c_void) -> u32 {
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventSynchronize(_: *mut c_void) -> u32 {
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamWaitEvent(_: *mut c_void, _: *mut c_void, _: u32) -> u32 {
    let result = status("cuStreamWaitEvent");
    if result == SUCCESS {
        WAITS.set(WAITS.get() + 1);
        completed("cuStreamWaitEvent");
    }
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuLaunchHostFunc(
    _: *mut c_void,
    callback: Option<unsafe extern "C" fn(*mut c_void)>,
    data: *mut c_void,
) -> u32 {
    let result = status("cuLaunchHostFunc");
    if result == SUCCESS {
        let callback = callback.expect("a host function");
        // SAFETY: the runtime's callback only reads its integer payload.
        unsafe { callback(data) };
        completed("cuLaunchHostFunc");
    }
    result
}
