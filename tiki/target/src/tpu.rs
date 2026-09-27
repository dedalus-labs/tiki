// Copyright © 2026 Dedalus Labs, Inc.

//! Google TPUs.

use crate::Target;

/// A Google TPU, named by its generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tpu {
    /// The generation, such as 7 for Ironwood.
    pub generation: u32,
}

/// Where a tensor's storage can live on a TPU.
///
/// The program decides how long data stays in each on-chip space, and asynchronous copies move
/// it between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpuMemory {
    /// High-bandwidth memory, off the core.
    Hbm,
    /// Vector memory, the large on-chip store the vector registers load from.
    Vector,
    /// Scalar memory, a small on-chip store for indices and control values.
    Scalar,
    /// The vector registers.
    Register,
}

/// The units that execute a TPU kernel, innermost first.
///
/// A core runs one program at a time, so a kernel's parallelism within a core is its vector
/// lanes and matrix unit, and across cores and chips it is explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpuUnit {
    /// One lane of a vector register.
    Lane,
    /// A core, which runs one program at a time.
    Core,
    Chip,
}

impl Target for Tpu {
    type Memory = TpuMemory;
    type Unit = TpuUnit;
    const NAME: &'static str = "tpu";
}
