// Copyright © 2026 Dedalus Labs, Inc.

//! Events: owned CUDA events without timing, destroyed on drop.

use super::context::with_device;
use super::device::Device;
use super::error::{CudaError, check};
use super::stream::Stream;
use super::sys;

/// A CUDA event on one device. It can be recorded only on that device's
/// streams, and any stream can wait on it.
#[derive(Debug)]
pub struct Event {
    /// `CUevent`, thread safe like every driver handle.
    handle: usize,
    device: Device,
}

impl Event {
    pub fn new(device: Device) -> Result<Event, CudaError> {
        let handle = with_device(device, || {
            let mut event = std::ptr::null_mut();
            let flags = sys::CUevent_flags::CU_EVENT_DISABLE_TIMING.0;
            // SAFETY: `event` outlives the call.
            check("cuEventCreate", unsafe { sys::cuEventCreate(&raw mut event, flags) })?;
            Ok(event as usize)
        })?;
        Ok(Event { handle, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub(crate) fn raw(&self) -> sys::CUevent {
        self.handle as sys::CUevent
    }

    /// Capture the work enqueued on `stream` so far. A record replaces the
    /// previous one; waits enqueued earlier keep the old one.
    pub fn record(&self, stream: &Stream) -> Result<(), CudaError> {
        if stream.device() != self.device {
            return Err(CudaError::raised(
                "cuEventRecord",
                sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
                format!(
                    "event on device {} recorded on a stream of device {}",
                    self.device.ordinal(),
                    stream.device().ordinal()
                ),
            ));
        }
        with_device(self.device, || {
            // SAFETY: both handles are live and belong to the same device.
            check("cuEventRecord", unsafe { sys::cuEventRecord(self.raw(), stream.raw()) })
        })
    }

    /// Whether the most recent record has completed. An event never recorded
    /// is complete.
    pub fn is_complete(&self) -> Result<bool, CudaError> {
        // SAFETY: the event is live while `self` is.
        match unsafe { sys::cuEventQuery(self.raw()) } {
            sys::CUresult::CUDA_SUCCESS => Ok(true),
            sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            code => check("cuEventQuery", code).map(|()| true),
        }
    }

    /// Block until the most recent record has completed.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        // SAFETY: the event is live while `self` is.
        check("cuEventSynchronize", unsafe { sys::cuEventSynchronize(self.raw()) })
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // Waits enqueued on the event keep their captured state. A failed
        // destroy leaks the event.
        let _ = with_device(self.device, || {
            // SAFETY: `self` owns the event and nothing uses it after drop.
            check("cuEventDestroy", unsafe { sys::cuEventDestroy_v2(self.raw()) })
        });
    }
}
