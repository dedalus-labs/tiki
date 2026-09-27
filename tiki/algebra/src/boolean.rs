// Copyright © 2026 Dedalus Labs, Inc.

//! Booleans under the boolean semiring, the algebra of masks and reachability.

use crate::operator::{And, Or};
use crate::structure::{CommutativeMonoid, Magma, Monoid, Semigroup, Semiring};

impl Magma<Or> for bool {
    fn operate(&self, other: &Self) -> Self {
        *self || *other
    }
}

impl Magma<And> for bool {
    fn operate(&self, other: &Self) -> Self {
        *self && *other
    }
}

impl Semigroup<Or> for bool {}
impl Semigroup<And> for bool {}

impl Monoid<Or> for bool {
    fn identity() -> Self {
        false
    }
}

impl Monoid<And> for bool {
    fn identity() -> Self {
        true
    }
}

impl CommutativeMonoid<Or> for bool {}
impl CommutativeMonoid<And> for bool {}

/// The boolean semiring, `(or, and)`.
impl Semiring<Or, And> for bool {}
