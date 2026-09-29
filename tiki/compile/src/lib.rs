// Copyright © 2026 Dedalus Labs, Inc.

//! Tiki's kernel compiler.
//!
//! A kernel that fails a proof never reaches lowering. This crate holds the
//! kernel IR and its proofs; lowering to LLVM IR builds on the [`Proof`] a
//! [`Partition`] returns. The compiler guide in `docs/src/dev/compiler.rst`
//! describes every stage.

mod partition;
mod proof;

pub use partition::Partition;
pub use proof::{Proof, ProofError};
