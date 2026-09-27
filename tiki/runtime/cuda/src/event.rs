// Copyright © 2026 Dedalus Labs, Inc.

//! RAII CUDA events drawn from a per-device pool.
//!
//! An event belongs to the device that created it and can only be recorded on
//! that device's streams. Dropping an [`Event`] returns it to the pool instead
//! of destroying it, so commit does not create and destroy an event per batch;
//! the pool lives for the process (cuEventCreate, cuEventRecord in cuda.h).

use std::sync::Mutex;

use crate::driver::{self, CudaError, EventHandle, Stream};

/// Free events per device index.
static POOL: Mutex<Vec<Vec<EventHandle>>> = Mutex::new(Vec::new());

/// A CUDA event created without timing, owned until drop returns it to the pool.
pub struct Event {
    /// CUevent, valid for the process lifetime.
    handle: EventHandle,
    /// Device the event was created on; it can only be recorded there.
    device: i32,
}

impl Event {
    /// Take a pooled event for `device`, creating one when the pool is empty.
    #[must_use = "an event that is never recorded orders nothing"]
    pub fn take(device: i32) -> Result<Self, CudaError> {
        let handle = match with_pool(device, Vec::pop) {
            Some(handle) => handle,
            None => driver::with_device(device, driver::event_create)?,
        };
        Ok(Self { handle, device })
    }

    pub fn record(&self, stream: Stream) -> Result<(), CudaError> {
        driver::with_device(self.device, || driver::event_record(self.handle, stream))
    }

    pub fn is_signaled(&self) -> Result<bool, CudaError> {
        driver::event_query(self.handle)
    }

    /// Make `stream` wait for the most recent record of this event.
    pub fn wait_on(&self, stream: Stream) -> Result<(), CudaError> {
        driver::stream_wait_event(stream, self.handle)
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        driver::event_synchronize(self.handle)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        with_pool(self.device, |pool| pool.push(self.handle));
    }
}

fn with_pool<T>(device: i32, f: impl FnOnce(&mut Vec<EventHandle>) -> T) -> T {
    let mut pools = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let index = device as usize;
    if pools.len() <= index {
        pools.resize_with(index + 1, Vec::new);
    }
    f(&mut pools[index])
}
