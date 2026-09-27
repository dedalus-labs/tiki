// Copyright © 2026 Dedalus Labs, Inc.

//! Streams: owned CUDA streams, destroyed on drop.

use std::ffi::c_void;

use super::context::with_device;
use super::device::Device;
use super::error::{CudaError, check};
use super::event::Event;
use super::sys;

/// Names a stream in maps and logs. It cannot reach the driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamId(usize);

/// A nonblocking CUDA stream on one device. Work enqueued on it runs in order.
#[derive(Debug)]
pub struct Stream {
    /// `CUstream`. The driver API is thread safe, so the stream may move
    /// between threads with its owner.
    handle: usize,
    device: Device,
}

impl Stream {
    pub fn new(device: Device) -> Result<Stream, CudaError> {
        let handle = with_device(device, || {
            let mut stream = std::ptr::null_mut();
            let flags = sys::CUstream_flags::CU_STREAM_NON_BLOCKING.0;
            // SAFETY: `stream` outlives the call.
            check("cuStreamCreate", unsafe { sys::cuStreamCreate(&raw mut stream, flags) })?;
            Ok(stream as usize)
        })?;
        Ok(Stream { handle, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn id(&self) -> StreamId {
        StreamId(self.handle)
    }

    pub(crate) fn raw(&self) -> sys::CUstream {
        self.handle as sys::CUstream
    }

    /// Block until everything enqueued so far has completed.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        // SAFETY: the stream is live while `self` is.
        check("cuStreamSynchronize", unsafe { sys::cuStreamSynchronize(self.raw()) })
    }

    /// Make later work on this stream wait for the most recent record of
    /// `event` at the time of this call. The event may belong to any device.
    pub fn wait(&self, event: &Event) -> Result<(), CudaError> {
        with_device(self.device, || {
            // SAFETY: both handles are live while their owners are.
            check("cuStreamWaitEvent", unsafe {
                sys::cuStreamWaitEvent(self.raw(), event.raw(), 0)
            })
        })
    }

    /// Run `callback(payload)` on a driver thread once the stream reaches
    /// this point. The callback must not call the driver.
    pub fn launch_host_func(
        &self,
        callback: extern "C" fn(*mut c_void),
        payload: usize,
    ) -> Result<(), CudaError> {
        with_device(self.device, || {
            // SAFETY: the driver passes `payload` back unchanged, and a safe
            // `extern "C" fn` is sound to call with any pointer value.
            check("cuLaunchHostFunc", unsafe {
                let callback = callback as unsafe extern "C" fn(*mut c_void);
                sys::cuLaunchHostFunc(self.raw(), Some(callback), payload as *mut c_void)
            })
        })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // Work still enqueued finishes first; the driver releases the stream
        // after it. A failed destroy leaks the stream and releases nothing early.
        let _ = with_device(self.device, || {
            // SAFETY: `self` owns the stream and nothing uses it after drop.
            check("cuStreamDestroy", unsafe { sys::cuStreamDestroy_v2(self.raw()) })
        });
    }
}
