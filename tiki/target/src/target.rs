// Copyright © 2026 Dedalus Labs, Inc.

//! The one trait every processor implements.

use std::fmt::Debug;

/// A family of parallel processors the kernel compiler can lower to.
///
/// SIMT processors, such as NVIDIA and AMD GPUs, run one program per thread across many
/// threads. A TPU core runs one program at a time and keeps its parallelism in vector lanes and
/// matrix units. Both are described by the same two associated types. A kernel's layouts map its
/// data onto [`Target::Memory`] and its work onto [`Target::Unit`], and the proofs of bounds,
/// disjoint writes and completion ordering hold on either.
pub trait Target {
    /// Where a tensor's storage can live. Each space has its own capacity, latency and set of
    /// units that can address it.
    type Memory: Copy + Eq + Debug;
    /// The units that execute a kernel's work, from the innermost, such as a thread or a vector
    /// lane, to the outermost, such as a grid or a chip.
    type Unit: Copy + Eq + Debug;
    /// The name shown in diagnostics.
    const NAME: &'static str;
}
