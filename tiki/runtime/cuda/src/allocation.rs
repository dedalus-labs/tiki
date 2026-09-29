// Copyright © 2026 Dedalus Labs, Inc.

//! The allocation a buffer holds for its lifetime.

use crate::driver::{Device, DeviceMemory};

/// Device memory rounded up to its size class. A buffer holds it for its
/// lifetime and returns it through [`Allocator::release`](crate::Allocator::release),
/// which caches it for reuse or frees it.
#[derive(Debug)]
pub struct Allocation {
    memory: DeviceMemory,
}

impl Allocation {
    pub(crate) fn new(memory: DeviceMemory) -> Self {
        Self { memory }
    }

    /// Rounded size in bytes.
    pub fn size(&self) -> usize {
        self.memory.len()
    }

    pub fn device(&self) -> Device {
        self.memory.device()
    }

    pub fn memory(&self) -> &DeviceMemory {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut DeviceMemory {
        &mut self.memory
    }

    pub(crate) fn into_memory(self) -> DeviceMemory {
        self.memory
    }
}
