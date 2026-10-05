// Copyright © 2026 Dedalus Labs, Inc.

//! Tiki's kernel compiler.
//!
//! A kernel that fails a proof never reaches lowering. This crate holds the
//! kernel IR, and each IR type proves its invariants when it is built: a
//! [`Partition`] exists only once its split of the domain is proven, so
//! lowering to LLVM IR takes the partition itself as its evidence. The
//! compiler guide in `docs/src/dev/compiler.rst` describes every stage.

mod error;
mod partition;

pub use error::PartitionError;
pub use partition::Partition;
