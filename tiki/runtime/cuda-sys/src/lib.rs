// Copyright © 2026 Dedalus Labs, Inc.

//! Tiki's only call site into the CUDA driver (`libcuda`), and the only Tiki
//! crate that contains `unsafe`. The declarations in `sys` are generated from
//! `cuda.h` by `bindings.sh`. The functions here check every result and hand
//! out integers (streams, events, pools, addresses), so no raw pointer or
//! foreign handle reaches the crates built on top.
//!
//! Every call that needs a context runs inside [`with_device`], which pushes
//! the device's primary context and pops it afterwards, so the caller's
//! current context is restored by construction, on success and on error.

use std::ffi::{CStr, c_char, c_void};
use std::fmt;
use std::sync::OnceLock;

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]
mod sys;

/// `CUstream` handle.
pub type Stream = usize;
/// Device, managed, or pinned host address in CUDA's unified address space.
pub type Address = usize;
/// Memory pool handle.
pub type MemPool = usize;
/// Event handle.
pub type EventHandle = usize;

// Device addresses are 64 bits; they are stored as usize.
const _: () = assert!(usize::BITS == 64);

/// A failed driver call, named so the caller can see which call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaError {
    pub call: &'static str,
    pub code: u32,
    pub name: String,
    pub description: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {}", self.call, self.description)
    }
}

impl std::error::Error for CudaError {}

fn error_text(
    code: sys::CUresult,
    lookup: unsafe extern "C" fn(sys::CUresult, *mut *const c_char) -> sys::CUresult,
) -> String {
    let mut text: *const c_char = std::ptr::null();
    // SAFETY: `text` outlives the call; on success it points to a static string.
    let found = unsafe { lookup(code, &mut text) };
    if found != sys::CUresult::CUDA_SUCCESS || text.is_null() {
        return format!("CUDA error {}", code.0);
    }
    // SAFETY: the driver returns a static, NUL-terminated string.
    unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
}

fn check(call: &'static str, code: sys::CUresult) -> Result<(), CudaError> {
    if code == sys::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    Err(CudaError {
        call,
        code: code.0,
        name: error_text(code, sys::cuGetErrorName),
        description: error_text(code, sys::cuGetErrorString),
    })
}

/// Primary context per device ordinal, retained once for the process.
static CONTEXTS: OnceLock<Vec<usize>> = OnceLock::new();

fn contexts() -> Result<&'static [usize], CudaError> {
    if let Some(contexts) = CONTEXTS.get() {
        return Ok(contexts);
    }
    // SAFETY: cuInit takes no pointers and is idempotent.
    check("cuInit", unsafe { sys::cuInit(0) })?;
    let mut retained = Vec::new();
    for ordinal in 0..device_count()? {
        let mut context = std::ptr::null_mut();
        // SAFETY: `context` outlives the call; primary contexts are never released.
        check("cuDevicePrimaryCtxRetain", unsafe {
            sys::cuDevicePrimaryCtxRetain(&mut context, device(ordinal)?)
        })?;
        retained.push(context as usize);
    }
    // A racing thread retains the same contexts; the driver counts the extra retains.
    Ok(CONTEXTS.get_or_init(|| retained))
}

fn device(ordinal: i32) -> Result<sys::CUdevice, CudaError> {
    let mut device = 0;
    // SAFETY: `device` outlives the call.
    check("cuDeviceGet", unsafe { sys::cuDeviceGet(&mut device, ordinal) })?;
    Ok(device)
}

pub fn device_count() -> Result<i32, CudaError> {
    // SAFETY: cuInit takes no pointers and is idempotent.
    check("cuInit", unsafe { sys::cuInit(0) })?;
    let mut count = 0;
    // SAFETY: `count` outlives the call.
    check("cuDeviceGetCount", unsafe { sys::cuDeviceGetCount(&mut count) })?;
    Ok(count)
}

/// Run `operation` with `device`'s primary context current, then restore the
/// caller's context. The caller's error wins; a failed pop means the driver
/// has already lost its context stack, and with it every resource.
pub fn with_device<T>(
    device: i32,
    operation: impl FnOnce() -> Result<T, CudaError>,
) -> Result<T, CudaError> {
    let contexts = contexts()?;
    let context = usize::try_from(device)
        .ok()
        .and_then(|index| contexts.get(index).copied())
        .ok_or_else(|| invalid_device(device))?;
    // SAFETY: `context` is a retained primary context.
    check("cuCtxPushCurrent", unsafe { sys::cuCtxPushCurrent_v2(context as sys::CUcontext) })?;
    let result = operation();
    let mut popped = std::ptr::null_mut();
    // SAFETY: `popped` outlives the call; this pops the context pushed above.
    let restored = check("cuCtxPopCurrent", unsafe { sys::cuCtxPopCurrent_v2(&mut popped) });
    let value = result?;
    restored?;
    Ok(value)
}

fn invalid_device(device: i32) -> CudaError {
    let code = sys::CUresult::CUDA_ERROR_INVALID_DEVICE;
    CudaError {
        call: "cuDeviceGet",
        code: code.0,
        name: error_text(code, sys::cuGetErrorName),
        description: format!("no CUDA device {device}"),
    }
}

fn attribute(ordinal: i32, attribute: sys::CUdevice_attribute) -> Result<i32, CudaError> {
    let mut value = 0;
    // SAFETY: `value` outlives the call.
    check("cuDeviceGetAttribute", unsafe {
        sys::cuDeviceGetAttribute(&mut value, attribute, device(ordinal)?)
    })?;
    Ok(value)
}

pub fn concurrent_managed_access(device: i32) -> Result<bool, CudaError> {
    let name = sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS;
    Ok(attribute(device, name)? != 0)
}

pub fn memory_pools_supported(device: i32) -> Result<bool, CudaError> {
    let name = sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED;
    Ok(attribute(device, name)? != 0)
}

pub fn default_mem_pool(device: i32) -> Result<MemPool, CudaError> {
    let mut pool = std::ptr::null_mut();
    // SAFETY: `pool` outlives the call.
    check("cuDeviceGetDefaultMemPool", unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool, self::device(device)?)
    })?;
    Ok(pool as MemPool)
}

pub fn mem_pool_reserved(pool: MemPool) -> Result<usize, CudaError> {
    let mut reserved: u64 = 0;
    let name = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT;
    // SAFETY: the attribute is a cuuint64_t and `reserved` outlives the call.
    check("cuMemPoolGetAttribute", unsafe {
        sys::cuMemPoolGetAttribute(pool as sys::CUmemoryPool, name, (&raw mut reserved).cast())
    })?;
    Ok(reserved as usize)
}

/// Total device memory of the current context's device.
pub fn total_memory() -> Result<usize, CudaError> {
    let (mut free, mut total) = (0, 0);
    // SAFETY: both outputs outlive the call.
    check("cuMemGetInfo", unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) })?;
    Ok(total)
}

pub fn create_stream() -> Result<Stream, CudaError> {
    let mut stream = std::ptr::null_mut();
    let flags = sys::CUstream_flags::CU_STREAM_NON_BLOCKING.0;
    // SAFETY: `stream` outlives the call.
    check("cuStreamCreate", unsafe { sys::cuStreamCreate(&mut stream, flags) })?;
    Ok(stream as Stream)
}

pub fn destroy_stream(stream: Stream) -> Result<(), CudaError> {
    // SAFETY: the caller owns `stream` and does not use it again.
    check("cuStreamDestroy", unsafe { sys::cuStreamDestroy_v2(stream as sys::CUstream) })
}

pub fn stream_synchronize(stream: Stream) -> Result<(), CudaError> {
    // SAFETY: `stream` is a live stream; the driver validates the handle.
    check("cuStreamSynchronize", unsafe { sys::cuStreamSynchronize(stream as sys::CUstream) })
}

pub fn malloc(size: usize) -> Result<Address, CudaError> {
    let mut address = 0;
    // SAFETY: `address` outlives the call.
    check("cuMemAlloc", unsafe { sys::cuMemAlloc_v2(&mut address, size) })?;
    Ok(address as Address)
}

pub fn malloc_async(size: usize, stream: Stream) -> Result<Address, CudaError> {
    let mut address = 0;
    // SAFETY: `address` outlives the call; `stream` is a live stream of the current device.
    check("cuMemAllocAsync", unsafe {
        sys::cuMemAllocAsync(&mut address, size, stream as sys::CUstream)
    })?;
    Ok(address as Address)
}

pub fn malloc_managed(size: usize) -> Result<Address, CudaError> {
    let mut address = 0;
    let flags = sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL.0;
    // SAFETY: `address` outlives the call.
    check("cuMemAllocManaged", unsafe { sys::cuMemAllocManaged(&mut address, size, flags) })?;
    Ok(address as Address)
}

pub fn malloc_host(size: usize) -> Result<Address, CudaError> {
    let mut address = std::ptr::null_mut();
    // SAFETY: `address` outlives the call.
    check("cuMemAllocHost", unsafe { sys::cuMemAllocHost_v2(&mut address, size) })?;
    Ok(address as Address)
}

/// Free memory from [`malloc`] or [`malloc_managed`]. The allocator frees each address once.
pub fn free(address: Address) -> Result<(), CudaError> {
    // SAFETY: the driver validates the address; no host memory is touched.
    check("cuMemFree", unsafe { sys::cuMemFree_v2(address as sys::CUdeviceptr) })
}

/// Free memory from [`malloc_async`] once `stream` reaches this point.
pub fn free_async(address: Address, stream: Stream) -> Result<(), CudaError> {
    // SAFETY: the driver validates the address and stream; no host memory is touched.
    check("cuMemFreeAsync", unsafe {
        sys::cuMemFreeAsync(address as sys::CUdeviceptr, stream as sys::CUstream)
    })
}

/// Free memory from [`malloc_host`].
pub fn free_host(address: Address) -> Result<(), CudaError> {
    // SAFETY: the address came from cuMemAllocHost and is freed once.
    check("cuMemFreeHost", unsafe { sys::cuMemFreeHost(address as *mut c_void) })
}

/// Enqueue a copy of `count` bytes between two unified addresses.
pub fn memcpy_async(
    dst: Address,
    src: Address,
    count: usize,
    stream: Stream,
) -> Result<(), CudaError> {
    // SAFETY: the driver resolves both addresses; the allocator keeps both ranges
    // live, and at least `count` bytes long, until the stream passes the copy.
    check("cuMemcpyAsync", unsafe {
        sys::cuMemcpyAsync(
            dst as sys::CUdeviceptr,
            src as sys::CUdeviceptr,
            count,
            stream as sys::CUstream,
        )
    })
}

/// Let `device` map a managed range without faulting.
pub fn advise_accessed_by(address: Address, count: usize, device: i32) -> Result<(), CudaError> {
    let location = sys::CUmemLocation {
        type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
        __bindgen_anon_1: sys::CUmemLocation_st__bindgen_ty_1 { id: device },
    };
    let advice = sys::CUmem_advise::CU_MEM_ADVISE_SET_ACCESSED_BY;
    // SAFETY: advice changes mapping policy only; the driver validates the range.
    check("cuMemAdvise", unsafe {
        sys::cuMemAdvise_v2(address as sys::CUdeviceptr, count, advice, location)
    })
}

pub fn event_create() -> Result<EventHandle, CudaError> {
    let mut event = std::ptr::null_mut();
    let flags = sys::CUevent_flags::CU_EVENT_DISABLE_TIMING.0;
    // SAFETY: `event` outlives the call; the event belongs to the current context.
    check("cuEventCreate", unsafe { sys::cuEventCreate(&mut event, flags) })?;
    Ok(event as EventHandle)
}

/// Capture the work enqueued on `stream` so far. A record replaces the
/// previous one; waits enqueued earlier keep the old one.
pub fn event_record(event: EventHandle, stream: Stream) -> Result<(), CudaError> {
    // SAFETY: both handles are live and belong to the same device.
    check("cuEventRecord", unsafe {
        sys::cuEventRecord(event as sys::CUevent, stream as sys::CUstream)
    })
}

/// Whether the most recent record of `event` has completed.
pub fn event_query(event: EventHandle) -> Result<bool, CudaError> {
    // SAFETY: `event` is a live event.
    match unsafe { sys::cuEventQuery(event as sys::CUevent) } {
        sys::CUresult::CUDA_SUCCESS => Ok(true),
        sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
        code => check("cuEventQuery", code).map(|()| true),
    }
}

pub fn event_synchronize(event: EventHandle) -> Result<(), CudaError> {
    // SAFETY: `event` is a live event.
    check("cuEventSynchronize", unsafe { sys::cuEventSynchronize(event as sys::CUevent) })
}

/// Make `stream` wait for the most recent record of `event` at call time.
pub fn stream_wait_event(stream: Stream, event: EventHandle) -> Result<(), CudaError> {
    // SAFETY: both handles are live; cross-device waits are permitted.
    check("cuStreamWaitEvent", unsafe {
        sys::cuStreamWaitEvent(stream as sys::CUstream, event as sys::CUevent, 0)
    })
}

/// Run `callback(payload)` on a driver thread once `stream` reaches this point.
/// The callback must not call the driver.
pub fn launch_host_func(
    stream: Stream,
    callback: extern "C" fn(*mut c_void),
    payload: usize,
) -> Result<(), CudaError> {
    // SAFETY: the payload is an integer the callback never dereferences.
    check("cuLaunchHostFunc", unsafe {
        let callback = callback as unsafe extern "C" fn(*mut c_void);
        sys::cuLaunchHostFunc(stream as sys::CUstream, Some(callback), payload as *mut c_void)
    })
}
