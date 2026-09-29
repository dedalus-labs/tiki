// Copyright © 2026 Dedalus Labs, Inc.

//! The allocator: size classes, a cache of released memory per device, and
//! the memory and cache limits.
//!
//! Released memory keeps the events of its last uses, so reusing it on any
//! stream waits for them, and freeing it follows them in stream order.
//! Evicted memory is dropped after the allocator lock is released, since a
//! drop enqueues a free on the driver.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::allocation::Allocation;
use crate::cache::SizeClassCache;
use crate::driver::{self, CudaError, Device, DeviceMemory, Stream};

pub const PAGE_SIZE: usize = 16384;
const SMALLEST_SIZE: usize = 8;
/// The allocator leaves one part in this many of device 0's memory free
/// before it frees cached memory: a 95% memory limit.
const FREE_PARTS: usize = 20;

struct State {
    memory_limit: usize,
    free_limit: usize,
    cache_limit: usize,
    active: usize,
    peak: usize,
    /// Released memory per device ordinal.
    caches: Vec<SizeClassCache<DeviceMemory>>,
}

impl State {
    fn cached_bytes(&self) -> usize {
        self.caches.iter().map(SizeClassCache::bytes).sum()
    }

    /// Remove at least `bytes` of cached memory, oldest first within each device.
    fn evict(&mut self, bytes: usize) -> Vec<DeviceMemory> {
        let mut evicted = Vec::new();
        let mut remaining = bytes;
        for cache in &mut self.caches {
            if remaining == 0 {
                break;
            }
            let before = cache.bytes();
            evicted.extend(cache.release(remaining));
            remaining = remaining.saturating_sub(before - cache.bytes());
        }
        evicted
    }
}

pub struct Allocator {
    devices: Vec<Device>,
    total_memory: usize,
    state: Mutex<State>,
}

fn round_size(size: usize) -> usize {
    if size <= SMALLEST_SIZE {
        SMALLEST_SIZE
    } else if size < PAGE_SIZE {
        size.next_power_of_two()
    } else {
        PAGE_SIZE * size.div_ceil(PAGE_SIZE)
    }
}

/// Position of a device in per-device tables.
fn slot(device: Device) -> usize {
    usize::try_from(device.ordinal()).expect("an opened ordinal is not negative")
}

impl Allocator {
    pub fn new() -> Result<Self, CudaError> {
        let devices =
            (0..driver::device_count()?).map(Device::open).collect::<Result<Vec<_>, _>>()?;
        let total_memory = Device::open(0)?.total_memory()?;
        let memory_limit = total_memory - total_memory / FREE_PARTS;
        let caches = devices.iter().map(|_| SizeClassCache::new(PAGE_SIZE)).collect();
        Ok(Self {
            devices,
            total_memory,
            state: Mutex::new(State {
                memory_limit,
                free_limit: total_memory - memory_limit,
                cache_limit: memory_limit,
                active: 0,
                peak: 0,
                caches,
            }),
        })
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Allocate at least `size` bytes on the device of `stream`, reusing
    /// released memory of a fitting size class when the cache holds one.
    pub fn allocate(&self, size: usize, stream: &Arc<Stream>) -> Result<Allocation, CudaError> {
        if size == 0 {
            return Ok(Allocation::new(DeviceMemory::allocate(stream, 0)?));
        }
        let size = round_size(size);
        let mut state = self.state();
        let reused = state.caches[slot(stream.device())].reuse(size);
        let (memory, mut evicted) = if let Some((_, memory)) = reused {
            (memory, Vec::new())
        } else {
            let pressure =
                (state.active + state.cached_bytes() + size).saturating_sub(state.memory_limit);
            let evicted = state.evict(pressure);
            drop(state);
            drop(evicted);
            let memory = DeviceMemory::allocate(stream, size)?;
            state = self.state();
            let evicted = self.relieve_pool_pressure(&mut state)?;
            (memory, evicted)
        };
        state.active += memory.len();
        state.peak = state.peak.max(state.active);
        let excess = state.cached_bytes().saturating_sub(state.cache_limit);
        evicted.extend(state.evict(excess));
        drop(state);
        drop(evicted);
        Ok(Allocation::new(memory))
    }

    /// Evict the free limit's worth of cached memory when a device's pool
    /// holds more than the memory the allocator leaves free.
    fn relieve_pool_pressure(&self, state: &mut State) -> Result<Vec<DeviceMemory>, CudaError> {
        if state.cached_bytes() == 0 {
            return Ok(Vec::new());
        }
        for device in &self.devices {
            if !device.memory_pools_supported() {
                continue;
            }
            if device.pool_reserved()? > self.total_memory - state.free_limit {
                let free_limit = state.free_limit;
                return Ok(state.evict(free_limit));
            }
        }
        Ok(Vec::new())
    }

    /// Return an allocation. Its memory is cached while the cache is below
    /// its limit and freed otherwise, after its last use.
    pub fn release(&self, allocation: Allocation) {
        let memory = allocation.into_memory();
        if memory.is_empty() {
            return;
        }
        let mut state = self.state();
        state.active -= memory.len();
        if state.cached_bytes() < state.cache_limit {
            state.caches[slot(memory.device())].recycle(memory.len(), memory);
        }
    }

    pub fn active_memory(&self) -> usize {
        self.state().active
    }

    pub fn peak_memory(&self) -> usize {
        self.state().peak
    }

    pub fn reset_peak_memory(&self) {
        self.state().peak = 0;
    }

    pub fn memory_limit(&self) -> usize {
        self.state().memory_limit
    }

    pub fn set_memory_limit(&self, limit: usize) -> usize {
        std::mem::replace(&mut self.state().memory_limit, limit)
    }

    pub fn cache_memory(&self) -> usize {
        self.state().cached_bytes()
    }

    pub fn set_cache_limit(&self, limit: usize) -> usize {
        std::mem::replace(&mut self.state().cache_limit, limit)
    }

    /// Free every cached allocation after its last use.
    pub fn clear_cache(&self) {
        let evicted: Vec<_> =
            self.state().caches.iter_mut().flat_map(SizeClassCache::clear).collect();
        drop(evicted);
    }
}

#[cfg(test)]
mod tests {
    use super::{PAGE_SIZE, round_size};

    // Invariant: sizes round to 8, then powers of two below a page, then whole pages.
    // Witness: 1 -> 8, 9 -> 16, 16383 -> 16384, 16385 -> 32768.
    #[test]
    fn rounding_matches_size_classes() {
        assert_eq!(round_size(1), 8);
        assert_eq!(round_size(8), 8);
        assert_eq!(round_size(9), 16);
        assert_eq!(round_size(PAGE_SIZE - 1), PAGE_SIZE);
        assert_eq!(round_size(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(round_size(PAGE_SIZE + 1), 2 * PAGE_SIZE);
    }
}
