// Copyright © 2026 Dedalus Labs, Inc.

//! Primary contexts, retained once per device for the process, and the push
//! and pop that make one current for a single operation.

use std::sync::OnceLock;

use super::device::Device;
use super::error::{CudaError, check};
use super::sys;

/// What the process retains for one device ordinal.
pub(crate) struct DeviceState {
    /// Primary context, never released.
    context: usize,
    /// Whether the device allocates in stream order from a memory pool.
    pub(crate) pools: bool,
}

static DEVICES: OnceLock<Vec<DeviceState>> = OnceLock::new();

/// Every device, with its primary context retained on first use.
pub(crate) fn devices() -> Result<&'static [DeviceState], CudaError> {
    if let Some(devices) = DEVICES.get() {
        return Ok(devices);
    }
    // SAFETY: cuInit takes no pointers and is idempotent.
    check("cuInit", unsafe { sys::cuInit(0) })?;
    let mut count = 0;
    // SAFETY: `count` outlives the call.
    check("cuDeviceGetCount", unsafe { sys::cuDeviceGetCount(&raw mut count) })?;
    let mut retained = Vec::new();
    for ordinal in 0..count {
        retained.push(retain(ordinal)?);
    }
    // A racing thread retains the same contexts; the driver counts the extra retains.
    Ok(DEVICES.get_or_init(|| retained))
}

fn retain(ordinal: i32) -> Result<DeviceState, CudaError> {
    let mut device = 0;
    // SAFETY: `device` outlives the call.
    check("cuDeviceGet", unsafe { sys::cuDeviceGet(&raw mut device, ordinal) })?;
    let mut context = std::ptr::null_mut();
    // SAFETY: `context` outlives the call; primary contexts are never released.
    check("cuDevicePrimaryCtxRetain", unsafe {
        sys::cuDevicePrimaryCtxRetain(&raw mut context, device)
    })?;
    let mut pools = 0;
    let attribute = sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED;
    // SAFETY: `pools` outlives the call.
    check("cuDeviceGetAttribute", unsafe {
        sys::cuDeviceGetAttribute(&raw mut pools, attribute, device)
    })?;
    Ok(DeviceState { context: context as usize, pools: pools != 0 })
}

/// Run `operation` with `device`'s primary context current, then restore the
/// caller's context, also when `operation` panics.
///
/// A failed restore stops the process: the thread's current context would
/// then be unknown, and later calls could run on another device than the
/// caller's.
pub(crate) fn with_device<T>(
    device: Device,
    operation: impl FnOnce() -> Result<T, CudaError>,
) -> Result<T, CudaError> {
    let context = device.state().context;
    // SAFETY: `context` is a retained primary context.
    check("cuCtxPushCurrent", unsafe { sys::cuCtxPushCurrent_v2(context as sys::CUcontext) })?;
    let _pushed = Pushed;
    operation()
}

/// Pops the pushed context when dropped, after the operation returns or
/// while it unwinds.
struct Pushed;

impl Drop for Pushed {
    fn drop(&mut self) {
        let mut popped = std::ptr::null_mut();
        // SAFETY: `popped` outlives the call; a context was pushed first.
        let restored =
            check("cuCtxPopCurrent", unsafe { sys::cuCtxPopCurrent_v2(&raw mut popped) });
        if let Err(error) = restored {
            eprintln!("tiki-cuda-sys: cannot restore the caller's CUDA context: {error}");
            std::process::abort();
        }
    }
}

impl Device {
    /// The retained state of a device, which exists once a `Device` does.
    pub(crate) fn state(self) -> &'static DeviceState {
        let devices = DEVICES.get().expect("a Device exists only after its context is retained");
        &devices[self.index()]
    }
}
