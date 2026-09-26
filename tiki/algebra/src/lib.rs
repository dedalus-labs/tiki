// Copyright © 2026 Dedalus Labs, Inc.

//! Algebraic structures as Rust traits.
//!
//! Tiki's kernels are folds and scans of monoids and matrix products over semirings, and its
//! layouts are linear maps into modules. This crate names those structures once so that every
//! kernel, layout and proof refers to the same laws.
//!
//! A structure is a set with operations that obey laws. Each trait adds one law to the trait
//! it extends:
//!
//! | Trait | Adds |
//! | --- | --- |
//! | [`Magma`] | a binary operation |
//! | [`Semigroup`] | associativity |
//! | [`Monoid`] | an identity |
//! | [`Group`] | an inverse for every element |
//! | [`CommutativeMonoid`] | commutativity |
//! | [`Semiring`] | a second monoid that distributes over the first |
//! | [`Ring`] | a semiring whose addition is a group |
//! | [`Field`] | a commutative ring with multiplicative inverses |
//! | [`Module`] | scaling of a commutative group by a ring |
//!
//! One type is often a monoid under several operations: a float is a monoid under `+`, under
//! `max` and under log-sum-exp. Each operation is therefore a marker type, as in the `alga`
//! crate, and a trait names the operation it is about: `Monoid<Max> for f32`.
//!
//! ```
//! use tiki_algebra::{Add, F2, LogSumExp, Max, Monoid, Mul};
//!
//! // The same float, three monoids.
//! assert_eq!(<f32 as Monoid<Add>>::identity(), 0.0);
//! assert_eq!(<f32 as Monoid<Max>>::identity(), f32::NEG_INFINITY);
//! assert_eq!(<f32 as Monoid<LogSumExp>>::identity(), f32::NEG_INFINITY);
//!
//! // In F2 addition is XOR, so every element is its own inverse.
//! assert_eq!(F2::ONE + F2::ONE, F2::ZERO);
//! assert_eq!(<F2 as Monoid<Mul>>::identity(), F2::ONE);
//! ```
//!
//! The traits assert their laws. [`laws`] checks them on values, and the crate's tests run
//! those checks on random inputs for every implementation. Floating-point addition is
//! associative only up to rounding, so the float implementations hold their laws up to
//! rounding.

#![forbid(unsafe_code)]

mod boolean;
mod f2;
mod float;
mod integer;
pub mod laws;
mod operator;
mod structure;

pub use f2::F2;
pub use operator::{Add, And, LogSumExp, Max, Min, Mul, Operator, Or};
pub use structure::{
    CommutativeMonoid, Field, Group, Magma, Module, Monoid, Ring, Semigroup, Semiring,
};
