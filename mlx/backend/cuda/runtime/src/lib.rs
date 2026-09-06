// Copyright © 2026 Dedalus Labs, Inc.

//! Rust-owned CUDA storage for Tiki's MLX backend.
//!
//! The allocator owns every device, unified, and small-pool allocation. Moving
//! storage to unified memory is one stream-ordered operation, so the device
//! source can never be released before the copy that reads it.

mod allocator;
mod bridge;
mod cache;
mod cudart;
mod pool;

pub use allocator::{init, runtime, AllocError, Allocation, Allocator};
pub use cudart::CudaError;
