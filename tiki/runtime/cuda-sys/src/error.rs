// Copyright © 2026 Dedalus Labs, Inc.

//! Errors: a failed driver call, and a copy its arguments rule out before any
//! driver call.

use std::ffi::{CStr, c_char};
use std::fmt;

use super::device::Device;
use super::sys;

/// A failed driver call, named so the caller can see which call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaError {
    pub call: &'static str,
    pub code: u32,
    pub name: String,
    pub description: String,
}

impl CudaError {
    /// An error the crate raises itself, with the driver's name for `code`.
    pub(crate) fn raised(call: &'static str, code: sys::CUresult, description: String) -> Self {
        CudaError { call, code: code.0, name: error_text(code, sys::cuGetErrorName), description }
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {}", self.call, self.description)
    }
}

impl std::error::Error for CudaError {}

/// A fill or copy that its arguments rule out before the driver is called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyError {
    /// The copy needs more bytes than the memory it reads or writes holds.
    Length {
        requested: usize,
        available: usize,
    },
    /// The memory belongs to another device than the stream.
    Device {
        stream: Device,
        memory: Device,
    },
    Cuda(CudaError),
}

impl From<CudaError> for CopyError {
    fn from(error: CudaError) -> Self {
        CopyError::Cuda(error)
    }
}

impl fmt::Display for CopyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CopyError::Length { requested, available } => {
                write!(f, "cannot copy {requested} bytes: the memory holds {available}")
            }
            CopyError::Device { stream, memory } => write!(
                f,
                "memory on device {} used on a stream of device {}",
                memory.ordinal(),
                stream.ordinal()
            ),
            CopyError::Cuda(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for CopyError {}

pub(crate) fn check(call: &'static str, code: sys::CUresult) -> Result<(), CudaError> {
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

fn error_text(
    code: sys::CUresult,
    lookup: unsafe extern "C" fn(sys::CUresult, *mut *const c_char) -> sys::CUresult,
) -> String {
    let mut text: *const c_char = std::ptr::null();
    // SAFETY: `text` outlives the call; on success it points to a static string.
    let found = unsafe { lookup(code, &raw mut text) };
    if found != sys::CUresult::CUDA_SUCCESS || text.is_null() {
        return format!("CUDA error {}", code.0);
    }
    // SAFETY: the driver returns a static, NUL-terminated string.
    unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
}
