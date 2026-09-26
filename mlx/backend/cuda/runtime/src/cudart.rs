// Copyright © 2026 Dedalus Labs, Inc.

//! The cudart symbols the allocator needs, behind checked safe wrappers.
//!
//! This is the crate's only unsafe layer. Every wrapper passes cudart
//! addresses that this crate obtained from cudart itself.

use std::ffi::{c_char, c_void, CStr};
use std::fmt;

pub type Stream = *mut c_void;
pub type MemPool = *mut c_void;

const SUCCESS: i32 = 0;
const STREAM_NON_BLOCKING: u32 = 0x01;
const MEM_ATTACH_GLOBAL: u32 = 0x01;
const MEMCPY_DEVICE_TO_HOST: i32 = 2;
const MEMCPY_DEFAULT: i32 = 4;
const MEM_ADVISE_SET_ACCESSED_BY: i32 = 5;
const MEM_LOCATION_TYPE_DEVICE: i32 = 1;
const MEM_POOL_ATTR_RESERVED_MEM_CURRENT: i32 = 5;
pub const DEV_ATTR_CONCURRENT_MANAGED_ACCESS: i32 = 89;
pub const DEV_ATTR_MEMORY_POOLS_SUPPORTED: i32 = 115;

#[repr(C)]
struct MemLocation {
    kind: i32,
    id: i32,
}

extern "C" {
    fn cudaGetErrorName(error: i32) -> *const c_char;
    fn cudaGetErrorString(error: i32) -> *const c_char;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaDeviceGetDefaultMemPool(pool: *mut MemPool, device: i32) -> i32;
    fn cudaMemPoolGetAttribute(pool: MemPool, attr: i32, value: *mut c_void) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut Stream, flags: u32) -> i32;
    fn cudaStreamSynchronize(stream: Stream) -> i32;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaMallocAsync(ptr: *mut *mut c_void, size: usize, stream: Stream) -> i32;
    fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaFreeAsync(ptr: *mut c_void, stream: Stream) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: Stream,
    ) -> i32;
    fn cudaMemAdvise(ptr: *const c_void, count: usize, advice: i32, location: MemLocation) -> i32;
}

/// A failed cudart call, named so the caller can see which primary failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaError {
    pub call: &'static str,
    pub code: i32,
    pub name: String,
    pub description: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {}", self.call, self.description)
    }
}

impl std::error::Error for CudaError {}

fn check(call: &'static str, code: i32) -> Result<(), CudaError> {
    if code == SUCCESS {
        return Ok(());
    }
    // SAFETY: cudart returns static strings for every error code.
    let (name, description) = unsafe {
        (
            CStr::from_ptr(cudaGetErrorName(code))
                .to_string_lossy()
                .into_owned(),
            CStr::from_ptr(cudaGetErrorString(code))
                .to_string_lossy()
                .into_owned(),
        )
    };
    Err(CudaError {
        call,
        code,
        name,
        description,
    })
}

pub fn device_count() -> Result<i32, CudaError> {
    let mut count = 0;
    // SAFETY: `count` outlives the call.
    check("cudaGetDeviceCount", unsafe {
        cudaGetDeviceCount(&mut count)
    })?;
    Ok(count)
}

pub fn set_device(device: i32) -> Result<(), CudaError> {
    // SAFETY: no pointers are involved.
    check("cudaSetDevice", unsafe { cudaSetDevice(device) })
}

pub fn device_attribute(device: i32, attr: i32) -> Result<i32, CudaError> {
    let mut value = 0;
    // SAFETY: `value` outlives the call.
    check("cudaDeviceGetAttribute", unsafe {
        cudaDeviceGetAttribute(&mut value, attr, device)
    })?;
    Ok(value)
}

pub fn default_mem_pool(device: i32) -> Result<MemPool, CudaError> {
    let mut pool = std::ptr::null_mut();
    // SAFETY: `pool` outlives the call.
    check("cudaDeviceGetDefaultMemPool", unsafe {
        cudaDeviceGetDefaultMemPool(&mut pool, device)
    })?;
    Ok(pool)
}

pub fn mem_pool_reserved(pool: MemPool) -> Result<usize, CudaError> {
    let mut reserved: usize = 0;
    // SAFETY: the attribute is a cuuint64_t and `reserved` outlives the call.
    check("cudaMemPoolGetAttribute", unsafe {
        cudaMemPoolGetAttribute(
            pool,
            MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
            (&mut reserved as *mut usize).cast(),
        )
    })?;
    Ok(reserved)
}

pub fn total_memory() -> Result<usize, CudaError> {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: both outputs outlive the call.
    check("cudaMemGetInfo", unsafe {
        cudaMemGetInfo(&mut free, &mut total)
    })?;
    Ok(total)
}

pub fn create_stream() -> Result<Stream, CudaError> {
    let mut stream = std::ptr::null_mut();
    // SAFETY: `stream` outlives the call.
    check("cudaStreamCreateWithFlags", unsafe {
        cudaStreamCreateWithFlags(&mut stream, STREAM_NON_BLOCKING)
    })?;
    Ok(stream)
}

pub fn stream_synchronize(stream: Stream) -> Result<(), CudaError> {
    // SAFETY: `stream` came from cudart and is never destroyed by this crate.
    check("cudaStreamSynchronize", unsafe {
        cudaStreamSynchronize(stream)
    })
}

fn checked_alloc(
    call: &'static str,
    code: i32,
    ptr: *mut c_void,
) -> Result<*mut c_void, CudaError> {
    check(call, code)?;
    Ok(ptr)
}

pub fn malloc(size: usize) -> Result<*mut c_void, CudaError> {
    let mut ptr = std::ptr::null_mut();
    // SAFETY: `ptr` outlives the call.
    checked_alloc("cudaMalloc", unsafe { cudaMalloc(&mut ptr, size) }, ptr)
}

pub fn malloc_async(size: usize, stream: Stream) -> Result<*mut c_void, CudaError> {
    let mut ptr = std::ptr::null_mut();
    // SAFETY: `ptr` outlives the call; `stream` is a live cudart stream.
    checked_alloc(
        "cudaMallocAsync",
        unsafe { cudaMallocAsync(&mut ptr, size, stream) },
        ptr,
    )
}

pub fn malloc_managed(size: usize) -> Result<*mut c_void, CudaError> {
    let mut ptr = std::ptr::null_mut();
    // SAFETY: `ptr` outlives the call.
    checked_alloc(
        "cudaMallocManaged",
        unsafe { cudaMallocManaged(&mut ptr, size, MEM_ATTACH_GLOBAL) },
        ptr,
    )
}

pub fn malloc_host(size: usize) -> Result<*mut c_void, CudaError> {
    let mut ptr = std::ptr::null_mut();
    // SAFETY: `ptr` outlives the call.
    checked_alloc(
        "cudaMallocHost",
        unsafe { cudaMallocHost(&mut ptr, size) },
        ptr,
    )
}

pub fn free(ptr: *mut c_void) -> Result<(), CudaError> {
    // SAFETY: `ptr` came from cudaMalloc or cudaMallocManaged and is freed once.
    check("cudaFree", unsafe { cudaFree(ptr) })
}

pub fn free_async(ptr: *mut c_void, stream: Stream) -> Result<(), CudaError> {
    // SAFETY: `ptr` came from cudaMallocAsync and is freed once.
    check("cudaFreeAsync", unsafe { cudaFreeAsync(ptr, stream) })
}

pub fn free_host(ptr: *mut c_void) -> Result<(), CudaError> {
    // SAFETY: `ptr` came from cudaMallocHost and is freed once.
    check("cudaFreeHost", unsafe { cudaFreeHost(ptr) })
}

/// Enqueue a copy of `count` bytes. `unified` selects cudaMemcpyDefault for
/// managed destinations; otherwise the destination is pinned host memory.
pub fn memcpy_async(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    unified: bool,
    stream: Stream,
) -> Result<(), CudaError> {
    let kind = if unified {
        MEMCPY_DEFAULT
    } else {
        MEMCPY_DEVICE_TO_HOST
    };
    // SAFETY: both ranges are live allocations of at least `count` bytes.
    check("cudaMemcpyAsync", unsafe {
        cudaMemcpyAsync(dst, src, count, kind, stream)
    })
}

pub fn advise_accessed_by(ptr: *const c_void, count: usize, device: i32) -> Result<(), CudaError> {
    let location = MemLocation {
        kind: MEM_LOCATION_TYPE_DEVICE,
        id: device,
    };
    // SAFETY: `ptr` is a live managed allocation of at least `count` bytes.
    check("cudaMemAdvise", unsafe {
        cudaMemAdvise(ptr, count, MEM_ADVISE_SET_ACCESSED_BY, location)
    })
}
