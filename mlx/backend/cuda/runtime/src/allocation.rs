// Copyright © 2026 Dedalus Labs, Inc.

//! The opaque allocation handle that C++ holds for the lifetime of a buffer.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::allocator::runtime;
use crate::cudart::{CudaError, Stream};

pub(crate) const EMPTY: Cached = Cached {
    kind: Kind::Empty,
    ptr: std::ptr::null_mut(),
};

/// Where an allocation's bytes live. Unified memory is managed or pinned host
/// memory that both the host and every device can address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Empty,
    Block { index: u32 },
    Unified,
    Device { device: i32 },
}

/// Storage held by the cache between release and reuse.
#[derive(Clone, Copy)]
pub(crate) struct Cached {
    pub(crate) kind: Kind,
    pub(crate) ptr: *mut c_void,
}

/// One owned allocation. C++ holds it as an opaque pointer for the lifetime of
/// the MLX buffer and returns it through `Allocator::release`. The address is
/// read for every kernel argument, so it lives in an atomic; the kind changes
/// only under the lock, together with the address, during migration.
pub struct Allocation {
    pub(crate) size: usize,
    pub(crate) ptr: AtomicUsize,
    pub(crate) kind: Mutex<Kind>,
}

// SAFETY: the struct holds cudart addresses, never host references, and cudart
// is thread-safe. Concurrent host access to the bytes is ordered by MLX.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    pub(crate) fn new(size: usize, cached: Cached) -> Self {
        Self {
            size,
            ptr: AtomicUsize::new(cached.ptr as usize),
            kind: Mutex::new(cached.kind),
        }
    }

    pub(crate) fn kind(&self) -> MutexGuard<'_, Kind> {
        self.kind.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Rounded size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// CUDA device holding the bytes, or -1 for unified memory.
    pub fn device(&self) -> i32 {
        match *self.kind() {
            Kind::Device { device } => device,
            _ => -1,
        }
    }

    /// Address usable by kernels, as an integer for the C++ bridge.
    pub fn data_ptr(&self) -> usize {
        self.ptr.load(Ordering::Acquire)
    }

    /// Address usable by the host. Device storage moves to unified memory first
    /// and the call returns after that copy completes.
    pub fn host_ptr(&self) -> Result<usize, CudaError> {
        runtime().migrate(self, None)?;
        Ok(self.data_ptr())
    }

    /// Move device storage to unified memory on `stream` without waiting.
    pub fn migrate_on(&self, stream: usize) -> Result<(), CudaError> {
        runtime().migrate(self, Some(stream as Stream))
    }
}
