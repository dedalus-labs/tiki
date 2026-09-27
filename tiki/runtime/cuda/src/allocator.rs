// Copyright © 2026 Dedalus Labs, Inc.

//! The allocator: size classes, small pool, cache, limits, and migration of
//! device storage to unified memory.

use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};

use crate::allocation::{Allocation, Cached, EMPTY, Kind};
use crate::cache::SizeClassCache;
use crate::driver::{self, Address, CudaError, MemPool, Stream};
use crate::pool::FreeList;
use crate::runtime::completion;

pub const PAGE_SIZE: usize = 16384;
const SMALL_BLOCK_SIZE: usize = 8;
const SMALL_POOL_SIZE: usize = 4 * PAGE_SIZE;
const SMALL_POOL_BLOCKS: u32 = (SMALL_POOL_SIZE / SMALL_BLOCK_SIZE) as u32;

#[derive(Debug)]
pub enum AllocError {
    Cuda(CudaError),
    InvalidDevice(i32),
    OutOfMemory { bytes: usize },
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(e) => e.fmt(f),
            Self::InvalidDevice(d) => write!(f, "[malloc] Invalid CUDA device {d}."),
            Self::OutOfMemory { bytes } => {
                write!(f, "[malloc] Unable to allocate {bytes} bytes.")
            }
        }
    }
}

impl std::error::Error for AllocError {}

impl From<CudaError> for AllocError {
    fn from(e: CudaError) -> Self {
        Self::Cuda(e)
    }
}

struct DeviceInfo {
    pool: Option<MemPool>,
    concurrent_managed_access: bool,
    /// Runtime-owned stream carrying every migration copy and device free.
    stream: Stream,
}

struct State {
    memory_limit: usize,
    free_limit: usize,
    max_pool_size: usize,
    active: usize,
    peak: usize,
    cache: SizeClassCache<Cached>,
    small: FreeList,
}

pub struct Allocator {
    devices: Vec<DeviceInfo>,
    managed: bool,
    total_memory: usize,
    small_base: Address,
    state: Mutex<State>,
}

fn round_size(size: usize) -> usize {
    if size <= SMALL_BLOCK_SIZE {
        SMALL_BLOCK_SIZE
    } else if size < PAGE_SIZE {
        size.next_power_of_two()
    } else {
        PAGE_SIZE * size.div_ceil(PAGE_SIZE)
    }
}

impl Allocator {
    pub fn new() -> Result<Self, CudaError> {
        let total_memory = driver::with_device(0, driver::total_memory)?;
        let devices = open_devices()?;
        let managed = devices.iter().all(|d| d.concurrent_managed_access);
        let small_base = match open_small_pool(managed, &devices) {
            Ok(base) => base,
            Err(error) => {
                close_devices(devices)?;
                return Err(error);
            }
        };
        let memory_limit = (total_memory as f64 * 0.95) as usize;
        Ok(Self {
            devices,
            managed,
            total_memory,
            small_base,
            state: Mutex::new(State {
                memory_limit,
                free_limit: total_memory - memory_limit,
                max_pool_size: memory_limit,
                active: 0,
                peak: 0,
                cache: SizeClassCache::new(PAGE_SIZE),
                small: FreeList::new(SMALL_POOL_BLOCKS),
            }),
        })
    }

    fn block_address(&self, index: u32) -> Address {
        self.small_base + index as usize * SMALL_BLOCK_SIZE
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn device_info(&self, device: i32) -> Result<&DeviceInfo, AllocError> {
        usize::try_from(device)
            .ok()
            .and_then(|i| self.devices.get(i))
            .ok_or(AllocError::InvalidDevice(device))
    }

    /// Allocate `size` bytes on `device` for use on `stream`. A null stream or a
    /// small request selects unified memory.
    pub fn allocate(
        &self,
        size: usize,
        device: i32,
        stream: Stream,
    ) -> Result<Allocation, AllocError> {
        if size == 0 {
            return Ok(Allocation::new(0, EMPTY));
        }
        let size = round_size(size);
        let device = if size <= SMALL_BLOCK_SIZE || stream == 0 { -1 } else { device };
        let mut state = self.state();
        let (size, cached) = match state.cache.reuse(size) {
            Some((capacity, cached)) => (capacity, cached.adopted_by(device, stream)),
            None => {
                let (guard, cached) = self.allocate_uncached(state, size, device, stream)?;
                state = guard;
                (size, cached)
            }
        };
        state.active += size;
        state.peak = state.peak.max(state.active);
        if state.cache.bytes() > state.max_pool_size {
            let excess = state.cache.bytes() - state.max_pool_size;
            self.release_cached(&mut state, excess)?;
        }
        drop(state);
        let allocation = Allocation::new(size, cached);
        match cached.kind {
            Kind::Device { device: held, .. } if held != device => {
                self.migrate(&allocation, (stream != 0).then_some(stream))?;
            }
            _ => {}
        }
        Ok(allocation)
    }

    /// Cache miss: relieve memory pressure, then take a small-pool block or
    /// fresh CUDA memory. The lock is released around the CUDA call.
    fn allocate_uncached<'a>(
        &'a self,
        mut state: MutexGuard<'a, State>,
        size: usize,
        device: i32,
        stream: Stream,
    ) -> Result<(MutexGuard<'a, State>, Cached), AllocError> {
        let pressure =
            (state.active + state.cache.bytes() + size) as i64 - state.memory_limit as i64;
        if pressure > 0 {
            self.release_cached(&mut state, pressure as usize)?;
        }
        let block = (size <= SMALL_BLOCK_SIZE).then(|| state.small.take()).flatten();
        let cached = match block {
            Some(index) => {
                Cached { kind: Kind::Block { index }, address: self.block_address(index) }
            }
            None => {
                drop(state);
                let cached = self.allocate_fresh(size, device, stream)?;
                state = self.state();
                cached
            }
        };
        if state.cache.bytes() > 0 {
            self.release_for_pool_pressure(&mut state)?;
        }
        Ok((state, cached))
    }

    fn allocate_fresh(
        &self,
        size: usize,
        device: i32,
        stream: Stream,
    ) -> Result<Cached, AllocError> {
        if device == -1 {
            let address = driver::with_device(0, || unified_malloc(self.managed, size))?;
            return Ok(Cached { kind: Kind::Unified, address });
        }
        let info = self.device_info(device)?;
        let address = driver::with_device(device, || match info.pool {
            Some(_) => driver::malloc_async(size, stream),
            None => driver::malloc(size),
        })?;
        if address == 0 {
            return Err(AllocError::OutOfMemory { bytes: size });
        }
        Ok(Cached { kind: Kind::Device { device, stream }, address })
    }

    /// Return an allocation. Storage is cached while the cache is below its
    /// limit and retired otherwise.
    pub fn release(&self, allocation: Allocation) -> Result<(), CudaError> {
        let cached = Cached {
            kind: allocation.kind.into_inner().unwrap_or_else(|e| e.into_inner()),
            address: allocation.address.into_inner(),
        };
        if cached.kind == Kind::Empty {
            return Ok(());
        }
        let mut state = self.state();
        state.active -= allocation.size;
        if state.cache.bytes() < state.max_pool_size {
            state.cache.recycle(allocation.size, cached);
            return Ok(());
        }
        let State { small, .. } = &mut *state;
        self.retire(small, cached)
    }

    fn retire(&self, small: &mut FreeList, cached: Cached) -> Result<(), CudaError> {
        match cached.kind {
            Kind::Empty => Ok(()),
            Kind::Block { index } => {
                small.give(index);
                Ok(())
            }
            Kind::Unified => driver::with_device(0, || unified_free(self.managed, cached.address)),
            Kind::Device { device, stream } => {
                let info = &self.devices[device as usize];
                driver::with_device(device, || match info.pool {
                    Some(_) => {
                        completion().order_after_latest(stream, info.stream)?;
                        driver::free_async(cached.address, info.stream)
                    }
                    None => driver::free(cached.address),
                })
            }
        }
    }

    fn release_cached(&self, state: &mut State, min_bytes: usize) -> Result<usize, CudaError> {
        let State { cache, small, .. } = &mut *state;
        let mut first_error = None;
        let count = cache.release(min_bytes, &mut |cached| {
            if let Err(e) = self.retire(small, cached) {
                first_error.get_or_insert(e);
            }
        });
        first_error.map_or(Ok(count), Err)
    }

    fn release_for_pool_pressure(&self, state: &mut State) -> Result<(), CudaError> {
        for info in &self.devices {
            let Some(pool) = info.pool else { continue };
            if driver::mem_pool_reserved(pool)? > self.total_memory - state.free_limit {
                let free_limit = state.free_limit;
                self.release_cached(state, free_limit)?;
                break;
            }
        }
        Ok(())
    }

    /// Move device storage to unified memory. The copy and the release of the
    /// device source are enqueued on one stream, so the source outlives the
    /// copy. Without a caller stream the call returns after the copy completes.
    pub(crate) fn migrate(
        &self,
        allocation: &Allocation,
        stream: Option<Stream>,
    ) -> Result<(), CudaError> {
        let mut kind = allocation.kind();
        let Kind::Device { device, stream: source } = *kind else {
            return Ok(());
        };
        let address = allocation.data_ptr();
        let info = &self.devices[device as usize];
        let blocking = stream.is_none() || info.pool.is_none();
        let stream = stream.unwrap_or(info.stream);
        driver::with_device(device, || {
            completion().order_after_latest(source, stream)?;
            let dst = unified_malloc(self.managed, allocation.size)?;
            let copied = driver::memcpy_async(dst, address, allocation.size, stream)
                .and_then(|()| if blocking { driver::stream_synchronize(stream) } else { Ok(()) });
            if let Err(e) = copied {
                unified_free(self.managed, dst)?;
                return Err(e);
            }
            match info.pool {
                Some(_) => driver::free_async(address, stream)?,
                None => driver::free(address)?,
            }
            allocation.address.store(dst, Ordering::Release);
            *kind = Kind::Unified;
            Ok(())
        })
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
        self.state().cache.bytes()
    }

    pub fn set_cache_limit(&self, limit: usize) -> usize {
        std::mem::replace(&mut self.state().max_pool_size, limit)
    }

    pub fn clear_cache(&self) -> Result<(), CudaError> {
        let mut state = self.state();
        let bytes = state.cache.bytes();
        self.release_cached(&mut state, bytes).map(|_| ())
    }
}

/// Create one runtime stream per device, destroying the earlier ones if a later device fails.
fn open_devices() -> Result<Vec<DeviceInfo>, CudaError> {
    let mut devices = Vec::new();
    for device in 0..driver::device_count()? {
        let opened = (|| {
            let concurrent_managed_access = driver::concurrent_managed_access(device)?;
            let pool = if driver::memory_pools_supported(device)? {
                Some(driver::default_mem_pool(device)?)
            } else {
                None
            };
            let stream = driver::with_device(device, driver::create_stream)?;
            Ok(DeviceInfo { pool, concurrent_managed_access, stream })
        })();
        match opened {
            Ok(info) => devices.push(info),
            Err(error) => {
                close_devices(devices)?;
                return Err(error);
            }
        }
    }
    Ok(devices)
}

fn close_devices(devices: Vec<DeviceInfo>) -> Result<(), CudaError> {
    for (device, info) in devices.into_iter().enumerate() {
        driver::with_device(device as i32, || driver::destroy_stream(info.stream))?;
    }
    Ok(())
}

/// The unified slab behind the small pool, readable by every device that can map it.
fn open_small_pool(managed: bool, devices: &[DeviceInfo]) -> Result<Address, CudaError> {
    driver::with_device(0, || {
        let base = unified_malloc(managed, SMALL_POOL_SIZE)?;
        let advised = devices
            .iter()
            .enumerate()
            .filter(|(_, info)| managed && info.concurrent_managed_access)
            .try_for_each(|(device, _)| {
                driver::advise_accessed_by(base, SMALL_POOL_SIZE, device as i32)
            });
        if let Err(error) = advised {
            unified_free(managed, base)?;
            return Err(error);
        }
        Ok(base)
    })
}

/// Managed memory when every device can access it concurrently, pinned host memory otherwise.
/// Call with a context current.
fn unified_malloc(managed: bool, size: usize) -> Result<Address, CudaError> {
    if managed { driver::malloc_managed(size) } else { driver::malloc_host(size) }
}

fn unified_free(managed: bool, address: Address) -> Result<(), CudaError> {
    if managed { driver::free(address) } else { driver::free_host(address) }
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
