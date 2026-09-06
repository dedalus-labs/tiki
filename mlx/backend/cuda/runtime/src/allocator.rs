// Copyright © 2026 Dedalus Labs, Inc.

//! The allocator: size classes, small pool, cache, limits, and migration of
//! device storage to unified memory.

use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::allocation::{Allocation, Cached, Kind, EMPTY};
use crate::cache::SizeClassCache;
use crate::cudart::{self, CudaError, MemPool, Stream};
use crate::pool::FreeList;

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
    small_base: *mut u8,
    state: Mutex<State>,
}

// SAFETY: cudart handles are process-global and cudart is thread-safe; all
// mutable bookkeeping is behind `state`.
unsafe impl Send for Allocator {}
unsafe impl Sync for Allocator {}

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
        let count = cudart::device_count()?;
        let mut devices = Vec::with_capacity(count as usize);
        for device in 0..count {
            let concurrent_managed_access =
                cudart::device_attribute(device, cudart::DEV_ATTR_CONCURRENT_MANAGED_ACCESS)? != 0;
            let pools =
                cudart::device_attribute(device, cudart::DEV_ATTR_MEMORY_POOLS_SUPPORTED)? != 0;
            cudart::set_device(device)?;
            devices.push(DeviceInfo {
                pool: pools
                    .then(|| cudart::default_mem_pool(device))
                    .transpose()?,
                concurrent_managed_access,
                stream: cudart::create_stream()?,
            });
        }
        cudart::set_device(0)?;
        let managed = devices.iter().all(|d| d.concurrent_managed_access);
        let total_memory = cudart::total_memory()?;
        let memory_limit = (total_memory as f64 * 0.95) as usize;
        let small_base = unified_malloc(managed, SMALL_POOL_SIZE)?.cast::<u8>();
        if managed {
            for (device, info) in devices.iter().enumerate() {
                if info.concurrent_managed_access {
                    cudart::advise_accessed_by(small_base.cast(), SMALL_POOL_SIZE, device as i32)?;
                }
            }
        }
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
        let device = if size <= SMALL_BLOCK_SIZE || stream.is_null() {
            -1
        } else {
            device
        };
        let mut state = self.state();
        let cached = match state.cache.reuse(size) {
            Some(cached) => cached,
            None => {
                let (guard, cached) = self.allocate_uncached(state, size, device, stream)?;
                state = guard;
                cached
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
        if let Kind::Device { device: held } = cached.kind {
            if held != device {
                self.migrate(&allocation, (!stream.is_null()).then_some(stream))?;
            }
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
        let block = (size <= SMALL_BLOCK_SIZE)
            .then(|| state.small.take())
            .flatten();
        let cached = match block {
            // SAFETY: `index` is below SMALL_POOL_BLOCKS, so the offset is inside the slab.
            Some(index) => Cached {
                kind: Kind::Block { index },
                ptr: unsafe { self.small_base.add(index as usize * SMALL_BLOCK_SIZE) }.cast(),
            },
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
            return Ok(Cached {
                kind: Kind::Unified,
                ptr: unified_malloc(self.managed, size)?,
            });
        }
        let info = self.device_info(device)?;
        cudart::set_device(device)?;
        let ptr = match info.pool {
            Some(_) => cudart::malloc_async(size, stream)?,
            None => cudart::malloc(size)?,
        };
        if ptr.is_null() {
            return Err(AllocError::OutOfMemory { bytes: size });
        }
        Ok(Cached {
            kind: Kind::Device { device },
            ptr,
        })
    }

    /// Return an allocation. Storage is cached while the cache is below its
    /// limit and retired otherwise.
    pub fn release(&self, allocation: Allocation) -> Result<(), CudaError> {
        let cached = Cached {
            kind: allocation
                .kind
                .into_inner()
                .unwrap_or_else(|e| e.into_inner()),
            ptr: allocation.ptr.into_inner() as *mut c_void,
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
            Kind::Unified => unified_free(self.managed, cached.ptr),
            Kind::Device { device } => {
                let info = &self.devices[device as usize];
                match info.pool {
                    Some(_) => cudart::free_async(cached.ptr, info.stream),
                    None => cudart::free(cached.ptr),
                }
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
            if cudart::mem_pool_reserved(pool)? > self.total_memory - state.free_limit {
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
        let Kind::Device { device } = *kind else {
            return Ok(());
        };
        let ptr = allocation.data_ptr() as *mut c_void;
        let info = &self.devices[device as usize];
        let blocking = stream.is_none() || info.pool.is_none();
        let stream = stream.unwrap_or(info.stream);
        let dst = unified_malloc(self.managed, allocation.size)?;
        let copied = cudart::set_device(device)
            .and_then(|()| cudart::memcpy_async(dst, ptr, allocation.size, self.managed, stream))
            .and_then(|()| {
                if blocking {
                    cudart::stream_synchronize(stream)
                } else {
                    Ok(())
                }
            });
        if let Err(e) = copied {
            unified_free(self.managed, dst)?;
            return Err(e);
        }
        match info.pool {
            Some(_) => cudart::free_async(ptr, stream)?,
            None => cudart::free(ptr)?,
        }
        allocation.ptr.store(dst as usize, Ordering::Release);
        *kind = Kind::Unified;
        Ok(())
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

fn unified_malloc(managed: bool, size: usize) -> Result<*mut c_void, CudaError> {
    if managed {
        cudart::malloc_managed(size)
    } else {
        cudart::malloc_host(size)
    }
}

fn unified_free(managed: bool, ptr: *mut c_void) -> Result<(), CudaError> {
    if managed {
        cudart::free(ptr)
    } else {
        cudart::free_host(ptr)
    }
}

static RUNTIME: OnceLock<Allocator> = OnceLock::new();

/// Construct the process-wide allocator. Later calls are no-ops.
pub fn init() -> Result<(), CudaError> {
    if RUNTIME.get().is_none() {
        let allocator = Allocator::new()?;
        let _ = RUNTIME.set(allocator);
    }
    Ok(())
}

pub fn runtime() -> &'static Allocator {
    RUNTIME
        .get()
        .expect("tiki-cuda-runtime: init() was not called")
}

#[cfg(test)]
mod tests {
    use super::{round_size, PAGE_SIZE};

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
