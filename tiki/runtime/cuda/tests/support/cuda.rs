// Copyright © 2026 Dedalus Labs, Inc.

//! A simulated CUDA driver: synchronous host memory, a thread-local context
//! stack, and streams and events that remember their device.

#![allow(non_snake_case)]

use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

const SUCCESS: u32 = 0;
const INVALID_DEVICE: u32 = 101;
const INJECTED: u32 = 999;
const DEVICES: i32 = 8;

static NEXT_HANDLE: AtomicUsize = AtomicUsize::new(1);
static ALLOCATIONS: AtomicI64 = AtomicI64::new(0);
static STREAMS: AtomicI64 = AtomicI64::new(0);
static EVENTS: AtomicI64 = AtomicI64::new(0);
static WAITS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Devices whose primary context is current, innermost last. The base
    /// entry is the thread's current device, as the CUDA runtime sets it.
    static STACK: RefCell<Vec<i32>> = RefCell::new(vec![0]);
    static FAILURE: Cell<&'static str> = const { Cell::new("") };
}

/// Device of the innermost current context.
pub fn current_device() -> i32 {
    STACK.with_borrow(|stack| *stack.last().expect("a context is current"))
}

/// Make `device` current on this thread, as a caller outside the runtime would.
pub fn set_current(device: i32) {
    STACK.with_borrow_mut(|stack| *stack.first_mut().expect("a base context") = device);
}

/// Make this thread's next call named `call` fail.
pub fn fail_next(call: &'static str) {
    FAILURE.set(call);
}

/// Live allocations, streams, and events.
pub fn live() -> (i64, i64, i64) {
    (
        ALLOCATIONS.load(Ordering::SeqCst),
        STREAMS.load(Ordering::SeqCst),
        EVENTS.load(Ordering::SeqCst),
    )
}

/// Stream waits enqueued so far.
pub fn waits() -> usize {
    WAITS.load(Ordering::SeqCst)
}

fn status(call: &str) -> u32 {
    if FAILURE.get() == call {
        FAILURE.set("");
        INJECTED
    } else {
        SUCCESS
    }
}

/// A new stream or event handle, with its device in the low byte.
fn handle_on(device: i32) -> *mut c_void {
    let serial = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    ((serial << 8) | (device as usize + 1)) as *mut c_void
}

fn handle_device(handle: *mut c_void) -> i32 {
    (handle as usize & 255) as i32 - 1
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
    // SAFETY: the adapter supplies live storage for this call.
    unsafe { *text = c"CUDA_ERROR_UNKNOWN".as_ptr() };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuGetErrorString(_: u32, text: *mut *const c_char) -> u32 {
    // SAFETY: the adapter supplies live storage for this call.
    unsafe { *text = c"injected CUDA error".as_ptr() };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDeviceGetCount(count: *mut i32) -> u32 {
    // SAFETY: the adapter supplies live storage for this call.
    unsafe { *count = DEVICES };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDeviceGet(device: *mut i32, ordinal: i32) -> u32 {
    if !(0..DEVICES).contains(&ordinal) {
        return INVALID_DEVICE;
    }
    // SAFETY: the adapter supplies live storage for this call.
    unsafe { *device = ordinal };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDeviceGetAttribute(value: *mut i32, _: u32, _: i32) -> u32 {
    // SAFETY: the adapter supplies live storage for this call.
    unsafe { *value = 1 };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDevicePrimaryCtxRetain(context: *mut *mut c_void, device: i32) -> u32 {
    // SAFETY: the adapter supplies live storage for this call.
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
        // SAFETY: the adapter supplies live storage for this call.
        unsafe { *context = (device as usize + 1) as *mut c_void };
    }
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuDeviceGetDefaultMemPool(pool: *mut *mut c_void, device: i32) -> u32 {
    // SAFETY: the adapter supplies live storage for this call.
    unsafe { *pool = (device as usize + 1) as *mut c_void };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemPoolGetAttribute(_: *mut c_void, _: u32, value: *mut u64) -> u32 {
    // SAFETY: the adapter supplies live storage for this call.
    unsafe { *value = 0 };
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> u32 {
    // SAFETY: the adapter supplies live storage for this call.
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
        // SAFETY: the adapter supplies live storage for this call.
        unsafe { *stream = handle_on(current_device()) };
        STREAMS.fetch_add(1, Ordering::SeqCst);
    }
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamDestroy_v2(_: *mut c_void) -> u32 {
    STREAMS.fetch_sub(1, Ordering::SeqCst);
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamSynchronize(_: *mut c_void) -> u32 {
    status("cuStreamSynchronize")
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamWaitEvent(_: *mut c_void, _: *mut c_void, _: u32) -> u32 {
    let result = status("cuStreamWaitEvent");
    if result == SUCCESS {
        WAITS.fetch_add(1, Ordering::SeqCst);
    }
    result
}

fn allocate(address: *mut u64, size: usize) -> u32 {
    // SAFETY: the adapter supplies live storage for this call.
    unsafe { *address = malloc(size) as u64 };
    ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
    SUCCESS
}

fn release(address: u64) -> u32 {
    // SAFETY: the address came from `allocate` and is freed once.
    unsafe { free(address as *mut c_void) };
    ALLOCATIONS.fetch_sub(1, Ordering::SeqCst);
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
    if handle_device(stream) != current_device() {
        return INVALID_DEVICE;
    }
    allocate(address, size)
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
unsafe extern "C" fn cuMemsetD8Async(address: u64, value: u8, size: usize, _: *mut c_void) -> u32 {
    let result = status("cuMemsetD8Async");
    if result == SUCCESS {
        // SAFETY: the address is a live simulated allocation of `size` bytes.
        unsafe { std::ptr::write_bytes(address as *mut u8, value, size) };
    }
    result
}

/// Every simulated copy is a host copy between live ranges of `size` bytes.
fn copy(call: &str, target: *mut c_void, source: *const c_void, size: usize) -> u32 {
    let result = status(call);
    if result == SUCCESS {
        // SAFETY: the adapter passes live, distinct ranges of `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(source.cast::<u8>(), target.cast::<u8>(), size) };
    }
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemcpyHtoDAsync_v2(
    target: u64,
    source: *const c_void,
    size: usize,
    _: *mut c_void,
) -> u32 {
    copy("cuMemcpyHtoDAsync", target as *mut c_void, source, size)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemcpyDtoHAsync_v2(
    target: *mut c_void,
    source: u64,
    size: usize,
    _: *mut c_void,
) -> u32 {
    copy("cuMemcpyDtoHAsync", target, source as *const c_void, size)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemcpyDtoDAsync_v2(
    target: u64,
    source: u64,
    size: usize,
    _: *mut c_void,
) -> u32 {
    copy("cuMemcpyDtoDAsync", target as *mut c_void, source as *const c_void, size)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventCreate(event: *mut *mut c_void, _: u32) -> u32 {
    let result = status("cuEventCreate");
    if result == SUCCESS {
        // SAFETY: the adapter supplies live storage for this call.
        unsafe { *event = handle_on(current_device()) };
        EVENTS.fetch_add(1, Ordering::SeqCst);
    }
    result
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventDestroy_v2(_: *mut c_void) -> u32 {
    EVENTS.fetch_sub(1, Ordering::SeqCst);
    SUCCESS
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> u32 {
    let result = status("cuEventRecord");
    if result != SUCCESS {
        return result;
    }
    let device = handle_device(event);
    if device != handle_device(stream) || device != current_device() {
        return INVALID_DEVICE;
    }
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
    }
    result
}
