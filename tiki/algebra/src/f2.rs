// Copyright © 2026 Dedalus Labs, Inc.

//! F2, the field of two elements.

use crate::operator::{Add, Mul};
use crate::structure::{
    CommutativeMonoid, Field, Group, Magma, Module, Monoid, Ring, Semigroup, Semiring,
};
use std::ops;

/// The field `{0, 1}`, whose addition is XOR and whose multiplication is AND.
///
/// A bit vector is a vector over F2, so XOR on an integer adds its bits in F2. A swizzle XORs
/// bit fields of an offset, so it is a linear map over F2, and a layout with XOR strides is a
/// layout whose codomain is a vector space over F2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct F2(bool);

impl F2 {
    pub const ZERO: F2 = F2(false);
    pub const ONE: F2 = F2(true);

    pub fn is_one(self) -> bool {
        self.0
    }
}

impl From<bool> for F2 {
    fn from(bit: bool) -> Self {
        F2(bit)
    }
}

impl ops::Add for F2 {
    type Output = F2;
    #[expect(clippy::suspicious_arithmetic_impl, reason = "addition in F2 is XOR")]
    fn add(self, other: F2) -> F2 {
        F2(self.0 ^ other.0)
    }
}

impl ops::Mul for F2 {
    type Output = F2;
    #[expect(clippy::suspicious_arithmetic_impl, reason = "multiplication in F2 is AND")]
    fn mul(self, other: F2) -> F2 {
        F2(self.0 & other.0)
    }
}

/// Every element of F2 is its own additive inverse.
impl ops::Neg for F2 {
    type Output = F2;
    fn neg(self) -> F2 {
        self
    }
}

impl Magma<Add> for F2 {
    fn operate(&self, other: &Self) -> Self {
        *self + *other
    }
}

impl Magma<Mul> for F2 {
    fn operate(&self, other: &Self) -> Self {
        *self * *other
    }
}

impl Semigroup<Add> for F2 {}
impl Semigroup<Mul> for F2 {}

impl Monoid<Add> for F2 {
    fn identity() -> Self {
        F2::ZERO
    }
}

impl Monoid<Mul> for F2 {
    fn identity() -> Self {
        F2::ONE
    }
}

impl CommutativeMonoid<Add> for F2 {}
impl CommutativeMonoid<Mul> for F2 {}

impl Group<Add> for F2 {
    fn inverse(&self) -> Self {
        -*self
    }
}

impl Semiring<Add, Mul> for F2 {}
impl Ring for F2 {}

impl Field for F2 {
    fn reciprocal(&self) -> Option<Self> {
        self.is_one().then_some(F2::ONE)
    }
}

/// A field is a module over itself.
impl Module<F2> for F2 {
    fn scale(&self, r: &F2) -> Self {
        *r * *self
    }
}
