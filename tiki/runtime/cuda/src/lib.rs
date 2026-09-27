// Copyright © 2026 Dedalus Labs, Inc.

//! Tiki's Rust CUDA runtime: device storage, streams, events, and completion.
//!
//! The allocator owns every device, unified, and small-pool allocation. The
//! completion runtime records an event at each commit and, once the device
//! signals it, runs the commit's [`Batch`] on its own worker thread. Moving
//! storage to unified memory and freeing it wait for the latest commit of the
//! allocation's assigned stream. Callers supply resource retention and
//! cross-stream dependencies; this is not per-kernel use tracking.
//!
//! Every driver call goes through `tiki-cuda-sys`; this crate contains no
//! `unsafe`.

#![forbid(unsafe_code)]

mod allocation;
mod allocator;
mod batch;
mod cache;
mod completion;
mod event;
mod pool;
mod runtime;

use tiki_cuda_sys as driver;

pub use allocation::Allocation;
pub use allocator::{AllocError, Allocator};
pub use batch::Batch;
pub use completion::Completion;
pub use event::Event;
pub use runtime::{allocator, completion, init};
pub use tiki_cuda_sys::{CudaError, Stream};
