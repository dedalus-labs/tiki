// Copyright © 2026 Dedalus Labs, Inc.

//! The integers, the ring of ordinary strides and coordinates.
//!
//! `i64` arithmetic panics on overflow in debug builds and wraps in release builds. Callers
//! that take extents from users check their ranges before they compute, as `tiki-cute` does.

use crate::operator::{Add, Max, Min, Mul};
use crate::structure::{
    CommutativeMonoid, Group, Magma, Module, Monoid, Ring, Semigroup, Semiring,
};

impl Magma<Add> for i64 {
    fn operate(&self, other: &Self) -> Self {
        self + other
    }
}

impl Magma<Mul> for i64 {
    fn operate(&self, other: &Self) -> Self {
        self * other
    }
}

impl Magma<Max> for i64 {
    fn operate(&self, other: &Self) -> Self {
        *self.max(other)
    }
}

impl Magma<Min> for i64 {
    fn operate(&self, other: &Self) -> Self {
        *self.min(other)
    }
}

impl Semigroup<Add> for i64 {}
impl Semigroup<Mul> for i64 {}
impl Semigroup<Max> for i64 {}
impl Semigroup<Min> for i64 {}

impl Monoid<Add> for i64 {
    fn identity() -> Self {
        0
    }
}

impl Monoid<Mul> for i64 {
    fn identity() -> Self {
        1
    }
}

impl Monoid<Max> for i64 {
    fn identity() -> Self {
        i64::MIN
    }
}

impl Monoid<Min> for i64 {
    fn identity() -> Self {
        i64::MAX
    }
}

impl CommutativeMonoid<Add> for i64 {}
impl CommutativeMonoid<Mul> for i64 {}
impl CommutativeMonoid<Max> for i64 {}
impl CommutativeMonoid<Min> for i64 {}

impl Group<Add> for i64 {
    fn inverse(&self) -> Self {
        -self
    }
}

impl Semiring<Add, Mul> for i64 {}
impl Ring for i64 {}

/// The ring is a module over itself.
impl Module<i64> for i64 {
    fn scale(&self, r: &i64) -> Self {
        r * self
    }
}
