// Copyright © 2026 Dedalus Labs, Inc.

//! Checked indexing transforms shared by Tiki layouts and CUDA schedules.
//! These values never own or dereference array storage.

#![deny(unsafe_code)]

mod error;
mod swizzle;

#[cfg(feature = "cxx-bridge")]
mod bridge;

pub use error::LayoutError;
pub use swizzle::Swizzle;
