// Copyright © 2026 Dedalus Labs, Inc.

//! The allocation handle a buffer holds for its lifetime.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::driver::{Address, CudaError, Stream};
use crate::runtime::allocator;

pub(crate) const EMPTY: Cached = Cached { kind: Kind::Empty, address: 0 };

/// Where an allocation's bytes live. Unified memory is managed or pinned host
/// memory that both the host and every device can address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Empty,
    Block {
        index: u32,
    },
    Unified,
    /// Device storage assigned for use on `stream`; commits on that stream
    /// are what the allocator orders frees and copies after.
    Device {
        device: i32,
        stream: Stream,
    },
}

/// Storage held by the cache between release and reuse.
#[derive(Clone, Copy)]
pub(crate) struct Cached {
    pub(crate) kind: Kind,
    pub(crate) address: Address,
}

impl Cached {
    /// Device storage reused on the same device now belongs to `stream`.
    pub(crate) fn adopted_by(self, device: i32, stream: Stream) -> Self {
        match self.kind {
            Kind::Device { device: held, .. } if held == device => {
                Self { kind: Kind::Device { device, stream }, address: self.address }
            }
            _ => self,
        }
    }
}

/// One owned allocation. A buffer holds it for its lifetime and returns it
/// through `Allocator::release`. The address is
/// read for every kernel argument, so it lives in an atomic; the kind changes
/// only under the lock, together with the address, during migration.
pub struct Allocation {
    pub(crate) size: usize,
    pub(crate) address: AtomicUsize,
    pub(crate) kind: Mutex<Kind>,
}

impl Allocation {
    pub(crate) fn new(size: usize, cached: Cached) -> Self {
        Self { size, address: AtomicUsize::new(cached.address), kind: Mutex::new(cached.kind) }
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
            Kind::Device { device, .. } => device,
            _ => -1,
        }
    }

    /// Address usable by kernels.
    pub fn data_ptr(&self) -> usize {
        self.address.load(Ordering::Acquire)
    }

    /// Address usable by the host. Device storage moves to unified memory first
    /// and the call returns after that copy completes.
    pub fn host_ptr(&self) -> Result<usize, CudaError> {
        allocator().migrate(self, None)?;
        Ok(self.data_ptr())
    }

    /// Move device storage to unified memory on `stream` without waiting.
    pub fn migrate_on(&self, stream: usize) -> Result<(), CudaError> {
        allocator().migrate(self, Some(stream))
    }
}
