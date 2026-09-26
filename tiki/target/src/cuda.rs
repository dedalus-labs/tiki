// Copyright © 2026 Dedalus Labs, Inc.

//! NVIDIA GPUs from Ampere onward.

use crate::Target;

/// An NVIDIA GPU, named by its compute capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cuda {
    /// The compute capability as PTX writes it, such as 80 for A100 and 100 for B200.
    pub sm: u32,
}

/// Where a tensor's storage can live on an NVIDIA GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaMemory {
    /// Device memory, visible to every thread.
    Global,
    /// On-chip memory shared by the threads of one block.
    Shared,
    /// Shared memory of another block in the same cluster, from Hopper onward.
    ClusterShared,
    /// Tensor memory, the accumulator store of `tcgen05` instructions, from Blackwell onward.
    Tensor,
    /// A thread's registers.
    Register,
}

/// The units that execute a CUDA kernel, innermost first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaUnit {
    Thread,
    /// 32 threads that execute in lockstep.
    Warp,
    /// Four warps that issue one `wgmma`, from Hopper onward.
    Warpgroup,
    Block,
    /// Blocks that share distributed shared memory, from Hopper onward.
    Cluster,
    Grid,
}

impl Target for Cuda {
    type Memory = CudaMemory;
    type Unit = CudaUnit;
    const NAME: &'static str = "cuda";
}
