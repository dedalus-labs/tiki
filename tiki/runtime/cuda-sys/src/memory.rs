// Copyright © 2026 Dedalus Labs, Inc.

//! Device memory: owned bytes on one device, and the copies between it and
//! host memory.
//!
//! Device memory remembers the stream of its last write and the streams that
//! read it since. A fill, copy, or read on another stream first waits for the
//! last writer, and a fill or copy into the memory also waits for its
//! readers. The wait goes through an event recorded at that moment on the
//! earlier stream, so work that stays on one stream records nothing, and
//! streams share memory without other ordering. Dropping memory frees it on
//! its allocation stream after every stream that used it.
//!
//! Copies to and from host memory wait for the device before they return, so
//! the driver never uses a borrowed host slice after its borrow ends.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::context::with_device;
use super::element::Element;
use super::error::{CopyError, CudaError, check};
use super::stream::Stream;
use super::sys;

/// Bytes of memory on one device, owned until drop.
///
/// Memory cannot be used after it is dropped:
///
/// ```compile_fail,E0382
/// # use std::sync::Arc;
/// # use tiki_cuda_sys::{Device, DeviceMemory, Stream};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let stream = Arc::new(Stream::new(Device::open(0)?)?);
/// let memory = DeviceMemory::allocate(&stream, 16)?;
/// drop(memory);
/// let mut host = [0u8; 16];
/// stream.read(&memory, &mut host)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct DeviceMemory {
    /// Device address, 0 when `len` is 0.
    address: sys::CUdeviceptr,
    len: usize,
    /// The stream the memory was allocated on and is freed on.
    stream: Arc<Stream>,
    uses: Mutex<Uses>,
}

/// The streams whose enqueued operations later operations must follow.
#[derive(Debug, Default)]
struct Uses {
    /// Stream of the last write, the allocation included.
    writer: Option<Arc<Stream>>,
    /// Other streams that read the memory since that write, each once.
    readers: Vec<Arc<Stream>>,
}

impl Uses {
    fn streams(&self) -> impl Iterator<Item = &Arc<Stream>> {
        self.writer.iter().chain(&self.readers)
    }

    fn written_by(stream: &Arc<Stream>) -> Uses {
        Uses { writer: Some(stream.clone()), readers: Vec::new() }
    }

    /// Add a reader, unless following another listed stream already covers
    /// it: a hand-off captures everything enqueued on its stream.
    fn add_reader(&mut self, stream: &Arc<Stream>) {
        if self.streams().all(|user| user.id() != stream.id()) {
            self.readers.push(stream.clone());
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Read,
    Write,
}

impl DeviceMemory {
    /// Allocate `len` bytes on the device of `stream`, in stream order on
    /// devices with memory pools. Zero bytes allocate nothing.
    pub fn allocate(stream: &Arc<Stream>, len: usize) -> Result<DeviceMemory, CudaError> {
        if len == 0 {
            return Ok(DeviceMemory::owning(0, 0, stream));
        }
        let device = stream.device();
        let pooled = device.memory_pools_supported();
        let address = with_device(device, || {
            let mut address = 0;
            if pooled {
                // SAFETY: `address` outlives the call and the stream is live.
                check("cuMemAllocAsync", unsafe {
                    sys::cuMemAllocAsync(&raw mut address, len, stream.raw())
                })?;
            } else {
                // SAFETY: `address` outlives the call.
                check("cuMemAlloc", unsafe { sys::cuMemAlloc_v2(&raw mut address, len) })?;
            }
            Ok(address)
        })?;
        let mut memory = DeviceMemory::owning(address, len, stream);
        if pooled {
            // A stream-ordered allocation is a write other streams must follow.
            *memory.uses_mut() = Uses::written_by(stream);
        }
        Ok(memory)
    }

    fn owning(address: sys::CUdeviceptr, len: usize, stream: &Arc<Stream>) -> DeviceMemory {
        DeviceMemory { address, len, stream: stream.clone(), uses: Mutex::default() }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> super::device::Device {
        self.stream.device()
    }

    fn uses(&self) -> MutexGuard<'_, Uses> {
        self.uses.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn uses_mut(&mut self) -> &mut Uses {
        self.uses.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let uses = std::mem::take(self.uses_mut());
        let (address, stream) = (self.address, &self.stream);
        let device = stream.device();
        // A failed free leaks the memory and releases nothing early.
        let _ = with_device(device, || {
            if device.memory_pools_supported() {
                for user in uses.streams() {
                    stream.wait_for(user)?;
                }
                // SAFETY: `self` owns the memory, and the free follows every
                // stream that used it.
                check("cuMemFreeAsync", unsafe { sys::cuMemFreeAsync(address, stream.raw()) })
            } else {
                for user in uses.streams() {
                    user.synchronize()?;
                }
                // SAFETY: `self` owns the memory, and every use has completed.
                check("cuMemFree", unsafe { sys::cuMemFree_v2(address) })
            }
        });
    }
}

impl Stream {
    /// Set every byte of `memory` to `value`, in stream order.
    pub fn fill(self: &Arc<Self>, memory: &mut DeviceMemory, value: u8) -> Result<(), CopyError> {
        self.check_device(memory)?;
        let (address, len) = (memory.address, memory.len);
        if len == 0 {
            return Ok(());
        }
        let uses = memory.uses_mut();
        self.follow(uses, Access::Write)?;
        with_device(self.device(), || {
            // SAFETY: the memory holds `len` bytes at `address` and outlives the
            // fill, since its drop follows this stream.
            check("cuMemsetD8Async", unsafe {
                sys::cuMemsetD8Async(address, value, len, self.raw())
            })
        })?;
        *uses = Uses::written_by(self);
        Ok(())
    }

    /// Copy all of `source` into the start of `target`, in stream order.
    pub fn copy(
        self: &Arc<Self>,
        source: &DeviceMemory,
        target: &mut DeviceMemory,
    ) -> Result<(), CopyError> {
        self.check_device(source)?;
        self.check_device(target)?;
        let (from, to, len) = (source.address, target.address, source.len);
        check_length(len, target.len)?;
        if len == 0 {
            return Ok(());
        }
        let mut source_uses = source.uses();
        let target_uses = target.uses_mut();
        self.follow(&source_uses, Access::Read)?;
        self.follow(target_uses, Access::Write)?;
        with_device(self.device(), || {
            // SAFETY: both memories hold at least `len` bytes, are distinct because
            // one is borrowed mutably, and outlive the copy, since their drops
            // follow this stream.
            check("cuMemcpyDtoDAsync", unsafe {
                sys::cuMemcpyDtoDAsync_v2(to, from, len, self.raw())
            })
        })?;
        source_uses.add_reader(self);
        *target_uses = Uses::written_by(self);
        Ok(())
    }

    /// Copy `source` from the host into the start of `target`. The copy has
    /// finished when the call returns, so `source` may change right after.
    pub fn write<T: Element>(
        &self,
        source: &[T],
        target: &mut DeviceMemory,
    ) -> Result<(), CopyError> {
        self.check_device(target)?;
        let (to, len) = (target.address, size_of_val(source));
        check_length(len, target.len)?;
        if len == 0 {
            return Ok(());
        }
        let uses = target.uses_mut();
        self.follow(uses, Access::Write)?;
        with_device(self.device(), || {
            // SAFETY: `source` is `len` readable bytes, borrowed until the call
            // returns, and the copy finishes first; the target holds `len` bytes.
            check("cuMemcpyHtoDAsync", unsafe {
                sys::cuMemcpyHtoDAsync_v2(to, source.as_ptr().cast(), len, self.raw())
            })
        })?;
        self.finish_host_copy();
        // The stream finished, and with it every use it followed.
        *uses = Uses::default();
        Ok(())
    }

    /// Copy the start of `source` into `target` on the host. The copy has
    /// finished when the call returns.
    ///
    /// The host slice is borrowed mutably, so a read cannot write through a
    /// shared borrow:
    ///
    /// ```compile_fail,E0596
    /// # use std::sync::Arc;
    /// # use tiki_cuda_sys::{Device, DeviceMemory, Stream};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stream = Arc::new(Stream::new(Device::open(0)?)?);
    /// let memory = DeviceMemory::allocate(&stream, 16)?;
    /// let frozen: Vec<u8> = vec![0; 16];
    /// stream.read(&memory, &mut frozen)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Element types have no invalid bit patterns, so `bool` is rejected:
    ///
    /// ```compile_fail,E0277
    /// # use std::sync::Arc;
    /// # use tiki_cuda_sys::{Device, DeviceMemory, Stream};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stream = Arc::new(Stream::new(Device::open(0)?)?);
    /// let memory = DeviceMemory::allocate(&stream, 16)?;
    /// let mut flags = [false; 16];
    /// stream.read(&memory, &mut flags)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read<T: Element>(
        &self,
        source: &DeviceMemory,
        target: &mut [T],
    ) -> Result<(), CopyError> {
        self.check_device(source)?;
        let (from, len) = (source.address, size_of_val(target));
        check_length(len, source.len)?;
        if len == 0 {
            return Ok(());
        }
        let uses = source.uses();
        self.follow(&uses, Access::Read)?;
        with_device(self.device(), || {
            // SAFETY: `target` is `len` writable bytes of a type with no invalid
            // bit patterns, borrowed until the call returns, and the copy
            // finishes first; the source holds `len` bytes.
            check("cuMemcpyDtoHAsync", unsafe {
                sys::cuMemcpyDtoHAsync_v2(target.as_mut_ptr().cast(), from, len, self.raw())
            })
        })?;
        self.finish_host_copy();
        Ok(())
    }

    fn check_device(&self, memory: &DeviceMemory) -> Result<(), CopyError> {
        let (stream, memory) = (self.device(), memory.device());
        if stream != memory {
            return Err(CopyError::Device { stream, memory });
        }
        Ok(())
    }

    /// Make this stream follow the memory's last writer, and before a write
    /// also its readers.
    fn follow(&self, uses: &Uses, access: Access) -> Result<(), CudaError> {
        if let Some(writer) = &uses.writer {
            self.wait_for(writer)?;
        }
        if access == Access::Write {
            for reader in &uses.readers {
                self.wait_for(reader)?;
            }
        }
        Ok(())
    }

    /// Wait for a copy that uses borrowed host memory. The borrow ends when
    /// the caller returns, so a copy that could still run after that must not
    /// be left behind: if the wait fails, the process stops.
    fn finish_host_copy(&self) {
        if let Err(error) = self.synchronize() {
            eprintln!("tiki-cuda-sys: a copy of host memory may still be running: {error}");
            std::process::abort();
        }
    }
}

fn check_length(requested: usize, available: usize) -> Result<(), CopyError> {
    if requested > available {
        return Err(CopyError::Length { requested, available });
    }
    Ok(())
}
