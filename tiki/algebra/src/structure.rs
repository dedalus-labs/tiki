// Copyright © 2026 Dedalus Labs, Inc.

//! The structure traits, each one law added to the trait it extends.

use crate::operator::{Add, Mul, Operator};

/// A set closed under the operation `O`.
pub trait Magma<O: Operator>: Sized + Clone {
    /// Returns `self ⊕ other`, where `⊕` is the operation `O`.
    #[must_use]
    fn operate(&self, other: &Self) -> Self;
}

/// A magma whose operation is associative: `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`.
///
/// Associativity is what lets threads combine partial results in any grouping, so every
/// reduction a kernel splits across threads needs it.
pub trait Semigroup<O: Operator>: Magma<O> {}

/// A semigroup with an identity: `e ⊕ a = a ⊕ e = a`.
///
/// The identity is the value a fold starts from and the value an empty segment folds to.
pub trait Monoid<O: Operator>: Semigroup<O> {
    /// Returns the identity element `e`.
    fn identity() -> Self;
}

/// A monoid whose operation is commutative: `a ⊕ b = b ⊕ a`.
///
/// Commutativity lets an atomic update combine contributions in whatever order they arrive.
pub trait CommutativeMonoid<O: Operator>: Monoid<O> {}

/// A monoid in which every element has an inverse: `a ⊕ a⁻¹ = e`.
pub trait Group<O: Operator>: Monoid<O> {
    /// Returns `a⁻¹`.
    #[must_use]
    fn inverse(&self) -> Self;
}

/// Two monoids, `⊕` named by `A` and `⊗` named by `M`, where `⊕` commutes, `⊗` distributes over
/// it, and the `⊕` identity annihilates: `e⊕ ⊗ a = a ⊗ e⊕ = e⊕`.
///
/// Matrix multiplication is defined over any semiring: `C[i, j] = ⊕ₖ A[i, k] ⊗ B[k, j]`.
pub trait Semiring<A: Operator, M: Operator>: CommutativeMonoid<A> + Monoid<M> {}

/// A semiring under `+` and `×` whose addition is a group, such as the integers.
pub trait Ring: Semiring<Add, Mul> + Group<Add> + CommutativeMonoid<Add> {}

/// A ring whose multiplication commutes and whose nonzero elements have multiplicative
/// inverses, such as [`crate::F2`].
pub trait Field: Ring + CommutativeMonoid<Mul> {
    /// Returns `a⁻¹` under multiplication, or `None` for zero.
    fn reciprocal(&self) -> Option<Self>;
}

/// A commutative group under addition that a ring `R` scales, with
/// `r ⊗ (a ⊕ b) = r ⊗ a ⊕ r ⊗ b` and `(r ⊕ s) ⊗ a = r ⊗ a ⊕ s ⊗ a`.
///
/// A layout's strides form a module over its coordinates' ring. The integers give ordinary
/// strides, and [`crate::F2`] gives XOR strides, a vector space since F2 is a field.
pub trait Module<R: Ring>: Group<Add> + CommutativeMonoid<Add> {
    /// Returns `r ⊗ self`.
    #[must_use]
    fn scale(&self, r: &R) -> Self;
}
