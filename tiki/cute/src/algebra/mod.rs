// Copyright © 2026 Dedalus Labs, Inc.

//! The layout algebra (Cecka, Section 3), as methods on [`crate::Layout`].
//!
//! Composition is the operation the others are built from:
//!
//! | Operation | Definition |
//! | --- | --- |
//! | [`Layout::coalesce`] | the flattest layout with the same function |
//! | [`Layout::compose`] | `A ∘ B`, evaluating `A` at every offset of `B` |
//! | [`Layout::complement`] | the modes that fill the gaps in `A`'s image |
//! | [`Layout::right_inverse`], [`Layout::left_inverse`] | partial inverses |
//! | [`Layout::logical_product`] | `(A, complement(A) ∘ B)`, one copy of `A` per coordinate of `B` |
//! | [`Layout::logical_divide`] | `A ∘ (B, complement(B))`, the tile and the tile index |
//! | [`Layout::recast`] | the same memory through a wider or narrower element |
//!
//! Every precondition is a [`crate::Truth`] checked with [`crate::Truth::require`], so an
//! operation on dynamic extents either proves its conditions or refuses.
//!
//! [`Layout::coalesce`]: crate::Layout::coalesce
//! [`Layout::compose`]: crate::Layout::compose
//! [`Layout::complement`]: crate::Layout::complement
//! [`Layout::right_inverse`]: crate::Layout::right_inverse
//! [`Layout::left_inverse`]: crate::Layout::left_inverse
//! [`Layout::logical_product`]: crate::Layout::logical_product
//! [`Layout::logical_divide`]: crate::Layout::logical_divide
//! [`Layout::recast`]: crate::Layout::recast

mod coalesce;
mod complement;
mod compose;
mod inverse;
mod order;
mod product;
mod recast;
mod sum;
