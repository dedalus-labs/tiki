// Copyright © 2026 Dedalus Labs, Inc.

//! The parallel processors Tiki's kernel compiler targets.
//!
//! A kernel is a function lowered onto one processor. Two facts about the processor decide how
//! the lowering goes: where a tensor's storage can live, and which units execute the work. A
//! [`Target`] names both, and nothing else, so the layout algebra and the kernel language stay
//! the same across processors while each target supplies its own memory spaces and units.
//!
//! [`Cuda`], [`Amd`] and [`Tpu`] are the targets Tiki plans for. Only their names exist yet: no
//! target lowers kernels. A researcher adds a processor by implementing [`Target`] for a new
//! type:
//!
//! ```
//! use tiki_target::Target;
//!
//! /// An accelerator whose cores share one scratchpad.
//! struct Scratchpad;
//!
//! #[derive(Debug, Clone, Copy, PartialEq, Eq)]
//! enum Memory { Dram, Scratchpad, Register }
//!
//! #[derive(Debug, Clone, Copy, PartialEq, Eq)]
//! enum Unit { Lane, Core, Chip }
//!
//! impl Target for Scratchpad {
//!     type Memory = Memory;
//!     type Unit = Unit;
//!     const NAME: &'static str = "scratchpad";
//! }
//! ```

#![forbid(unsafe_code)]

mod amd;
mod cuda;
mod target;
mod tpu;

pub use amd::{Amd, AmdMemory, AmdUnit};
pub use cuda::{Cuda, CudaMemory, CudaUnit};
pub use target::Target;
pub use tpu::{Tpu, TpuMemory, TpuUnit};
