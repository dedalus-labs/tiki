// Copyright © 2026 Dedalus Labs, Inc.

//! Tiki's Rust CUDA runtime: cached device memory and completion.
//!
//! The allocator hands out device memory by size class and caches released
//! memory for reuse. The completion runtime records an event at each commit
//! and, once the device signals it, runs the commit's [`Batch`] on its own
//! worker thread. Device memory orders its own uses across streams, so
//! callers supply only the values a commit retains.
//!
//! Every driver call goes through `tiki-cuda-sys`; this crate contains no
//! `unsafe`.

#![forbid(unsafe_code)]

mod allocation;
mod allocator;
mod batch;
mod cache;
mod completion;
mod runtime;

use tiki_cuda_sys as driver;

pub use allocation::Allocation;
pub use allocator::Allocator;
pub use batch::Batch;
pub use completion::Completion;
pub use runtime::{allocator, completion, init};
pub use tiki_cuda_sys::{CopyError, CudaError, Device, DeviceMemory, Element, Stream, StreamId};
