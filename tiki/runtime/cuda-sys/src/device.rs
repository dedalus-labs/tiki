// Copyright © 2026 Dedalus Labs, Inc.

//! Devices: checked ordinals whose primary context the process retains.

use super::context::{devices, with_device};
use super::error::{CudaError, check};
use super::sys;

/// A CUDA device. Opening checks the ordinal, so every `Device` names one the
/// process can use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Device {
    ordinal: i32,
}

/// Number of CUDA devices.
pub fn device_count() -> Result<i32, CudaError> {
    Ok(i32::try_from(devices()?.len()).expect("the driver counts devices in an i32"))
}

impl Device {
    pub fn open(ordinal: i32) -> Result<Device, CudaError> {
        let count = devices()?.len();
        match usize::try_from(ordinal) {
            Ok(index) if index < count => Ok(Device { ordinal }),
            _ => Err(CudaError::raised(
                "cuDeviceGet",
                sys::CUresult::CUDA_ERROR_INVALID_DEVICE,
                format!("no CUDA device {ordinal}"),
            )),
        }
    }

    pub fn ordinal(self) -> i32 {
        self.ordinal
    }

    /// Position in the process's device table; `open` checked it is in range.
    pub(crate) fn index(self) -> usize {
        usize::try_from(self.ordinal).expect("an opened ordinal is not negative")
    }

    /// Whether the device allocates in stream order from a memory pool.
    pub fn memory_pools_supported(self) -> bool {
        self.state().pools
    }

    /// Total memory of the device in bytes.
    pub fn total_memory(self) -> Result<usize, CudaError> {
        with_device(self, || {
            let (mut free, mut total) = (0, 0);
            // SAFETY: both outputs outlive the call.
            check("cuMemGetInfo", unsafe { sys::cuMemGetInfo_v2(&raw mut free, &raw mut total) })?;
            Ok(total)
        })
    }

    /// Bytes the default memory pool holds from the driver, including freed
    /// bytes it keeps for reuse.
    pub fn pool_reserved(self) -> Result<usize, CudaError> {
        let mut device = 0;
        // SAFETY: `device` outlives the call.
        check("cuDeviceGet", unsafe { sys::cuDeviceGet(&raw mut device, self.ordinal) })?;
        let mut pool = std::ptr::null_mut();
        // SAFETY: `pool` outlives the call.
        check("cuDeviceGetDefaultMemPool", unsafe {
            sys::cuDeviceGetDefaultMemPool(&raw mut pool, device)
        })?;
        let mut reserved: u64 = 0;
        let name = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT;
        // SAFETY: the attribute is a cuuint64_t and `reserved` outlives the call.
        check("cuMemPoolGetAttribute", unsafe {
            sys::cuMemPoolGetAttribute(pool, name, (&raw mut reserved).cast())
        })?;
        Ok(usize::try_from(reserved).expect("reserved bytes fit in usize"))
    }
}
