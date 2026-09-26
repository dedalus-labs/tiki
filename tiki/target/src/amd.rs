// Copyright © 2026 Dedalus Labs, Inc.

//! AMD GPUs.

use crate::Target;

/// An AMD GPU, named by its LLVM processor name, such as `gfx942` for MI300.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Amd {
    /// The processor name LLVM's AMDGPU backend takes.
    pub processor: &'static str,
}

/// Where a tensor's storage can live on an AMD GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmdMemory {
    /// Device memory, visible to every lane.
    Global,
    /// The local data share, on-chip memory shared by one workgroup.
    Local,
    /// A lane's vector registers.
    Register,
}

/// The units that execute an AMD kernel, innermost first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmdUnit {
    Lane,
    /// 32 or 64 lanes that execute in lockstep, depending on the processor.
    Wavefront,
    Workgroup,
    Grid,
}

impl Target for Amd {
    type Memory = AmdMemory;
    type Unit = AmdUnit;
    const NAME: &'static str = "amd";
}
