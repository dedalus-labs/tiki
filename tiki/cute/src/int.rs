// Copyright © 2026 Dedalus Labs, Inc.

//! Extents, strides and coordinates: integers known at compile time or fixed at launch.
//!
//! CuTe keeps two kinds of integer. A static integer is a compile-time constant, and every
//! layout operation on it folds away. A dynamic integer is a value the kernel reads at launch,
//! such as a matrix's row count. Tiki represents a dynamic integer as a polynomial over launch
//! parameters ([`Param`]), so the compiler can still compute with it: `4 * N` is a value, and
//! so is `ceil(N / 128)`.
//!
//! Arithmetic never fails for want of facts. Questions do: whether `N` is a multiple of 16
//! depends on the launch unless `N`'s divisor settles it. Each predicate returns a [`Truth`],
//! and the algebra decides at each call site what an open answer means.

use crate::param::Param;
use crate::poly::{Atom, OVERFLOW, Poly, floor_div};
use crate::truth::Truth;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

/// An integer the layout algebra computes with.
///
/// Constant polynomials are always `Static`, so two `Int`s are structurally equal exactly when
/// they are equal for every launch. Arithmetic panics on `i64` overflow, which only a layout
/// spanning more than 2^63 elements reaches.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Int {
    /// A compile-time constant.
    Static(i64),
    /// A polynomial over launch parameters with at least one non-constant term.
    Dynamic(Arc<Poly>),
}

impl Int {
    pub fn as_static(&self) -> Option<i64> {
        match self {
            Int::Static(value) => Some(*value),
            Int::Dynamic(_) => None,
        }
    }

    pub fn is_static(&self) -> bool {
        matches!(self, Int::Static(_))
    }

    /// Returns whether this is 0 for every launch. Equal for every launch means structurally
    /// equal, so no facts are consulted.
    pub fn is_zero(&self) -> bool {
        *self == Int::Static(0)
    }

    /// Returns whether this is 1 for every launch.
    pub fn is_one(&self) -> bool {
        *self == Int::Static(1)
    }

    /// Returns `floor(self / divisor)`, as Python's `//`. The quotient is a polynomial when the
    /// divisor provably divides every term, and an opaque `floor` atom otherwise.
    ///
    /// # Panics
    ///
    /// Panics when both are static and `divisor` is 0. Every divisor the algebra passes is an
    /// extent or a nonzero stride coefficient.
    #[must_use]
    pub fn div_floor(&self, divisor: &Int) -> Int {
        if let (Int::Static(a), Int::Static(b)) = (self, divisor) {
            return Int::Static(floor_div(*a, *b));
        }
        let (a, b) = (self.poly(), divisor.poly());
        if let Some(quotient) = a.exact_div(&b) {
            return Int::from_poly(quotient);
        }
        // A static divisor splits the dividend into the terms it provably divides and a rest.
        // When the rest lies in `0..divisor` for every launch it adds nothing to the floor, so
        // `ceil(M / 128)`, written `floor((M + 127) / 128)`, is `floor(M / 128)` when 128
        // divides `M`. Both spell the same exact quotient atom and compare equal.
        if let Some(c) = divisor.as_static().filter(|&c| c > 0) {
            let (multiples, rest) = a.split_multiples(c);
            let (low, high) = rest.bounds();
            let vanishes =
                low.is_some_and(|low| low >= 0) && high.is_some_and(|high| high < c.into());
            if vanishes && multiples != a {
                return Int::from_poly(multiples).div_floor(divisor);
            }
        }
        Int::from_poly(Poly::atom(Atom::Floor(Arc::new(a), Arc::new(b))))
    }

    /// Returns `self - divisor * floor(self / divisor)`, as Python's `%`. It is the static 0
    /// whenever divisibility is provable, even when the quotient is opaque.
    #[must_use]
    pub fn rem_floor(&self, divisor: &Int) -> Int {
        if let (Int::Static(a), Int::Static(b)) = (self, divisor) {
            return Int::Static(a - b * floor_div(*a, *b));
        }
        if self.is_multiple_of(divisor).is_proven() {
            return Int::Static(0);
        }
        self - &(divisor * &self.div_floor(divisor))
    }

    /// Returns `ceil(self / divisor)` for a positive divisor.
    #[must_use]
    pub fn div_ceil(&self, divisor: &Int) -> Int {
        (&(self + divisor) - &Int::Static(1)).div_floor(divisor)
    }

    /// Returns whether `divisor` divides this for every launch.
    ///
    /// A static divisor is proven from the coefficients and the parameters' divisors, so `8*N`
    /// is a multiple of 16 when `N` is a multiple of 2. A dynamic divisor is proven only by
    /// exact polynomial division, so `N*M` is a multiple of `N`.
    #[must_use]
    pub fn is_multiple_of(&self, divisor: &Int) -> Truth {
        match (self, divisor) {
            (Int::Static(a), Int::Static(b)) => Truth::from(*b != 0 && a % b == 0),
            (_, Int::Static(b)) if *b != 0 && self.poly().known_multiple() % b == 0 => {
                Truth::Proven
            }
            _ if self.poly().exact_div(&divisor.poly()).is_some() => Truth::Proven,
            _ => Truth::Open,
        }
    }

    #[must_use]
    pub fn is_positive(&self) -> Truth {
        let (low, high) = self.range();
        truth(low.is_some_and(|low| low > 0), high.is_some_and(|high| high <= 0))
    }

    #[must_use]
    pub fn equals(&self, other: &Int) -> Truth {
        let difference = self - other;
        let (low, high) = difference.range();
        let apart = low.is_some_and(|low| low > 0) || high.is_some_and(|high| high < 0);
        truth(difference.is_zero(), apart)
    }

    /// Returns whether `self >= other` for every launch.
    #[must_use]
    pub fn is_at_least(&self, other: &Int) -> Truth {
        let (low, high) = (self - other).range();
        truth(low.is_some_and(|low| low >= 0), high.is_some_and(|high| high < 0))
    }

    /// Returns the order of `self` and `other` when it is the same for every launch.
    pub(crate) fn compare(&self, other: &Int) -> Option<Ordering> {
        (self - other).sign()
    }

    /// Returns the value under a concrete assignment of every parameter.
    pub fn eval(&self, value: &impl Fn(&Param) -> i64) -> i64 {
        match self {
            Int::Static(v) => *v,
            Int::Dynamic(poly) => poly.eval(value),
        }
    }

    /// Returns the sign shared by every launch, from the polynomial's bounds.
    fn sign(&self) -> Option<Ordering> {
        match self.range() {
            (Some(low), Some(high)) if low == high => Some(low.cmp(&0)),
            (Some(low), _) if low > 0 => Some(Ordering::Greater),
            (_, Some(high)) if high < 0 => Some(Ordering::Less),
            _ => None,
        }
    }

    /// Returns the least and greatest value over every admitted launch, `None` where unbounded.
    fn range(&self) -> (Option<i128>, Option<i128>) {
        match self {
            Int::Static(value) => (Some((*value).into()), Some((*value).into())),
            Int::Dynamic(poly) => poly.bounds(),
        }
    }

    fn from_poly(poly: Poly) -> Int {
        match poly.as_constant() {
            Some(value) => Int::Static(value),
            None => Int::Dynamic(Arc::new(poly)),
        }
    }

    fn poly(&self) -> Poly {
        match self {
            Int::Static(value) => Poly::constant(*value),
            Int::Dynamic(poly) => (**poly).clone(),
        }
    }
}

/// Returns `Proven` when `holds` is shown, `Refuted` when `fails` is shown, and `Open` otherwise.
fn truth(holds: bool, fails: bool) -> Truth {
    match (holds, fails) {
        (true, _) => Truth::Proven,
        (false, true) => Truth::Refuted,
        (false, false) => Truth::Open,
    }
}

/// Implements an operator for owned and borrowed operands.
///
/// PERF: two static operands take the `i64` path and never build a polynomial. Almost every
/// operation in a kernel's tile layouts is static, so the polynomial path runs only for the
/// few layouts that carry launch parameters.
macro_rules! arithmetic {
    ($trait:ident, $method:ident, $checked:ident, $poly:expr) => {
        impl $trait<&Int> for &Int {
            type Output = Int;
            fn $method(self, rhs: &Int) -> Int {
                match (self, rhs) {
                    (Int::Static(a), Int::Static(b)) => {
                        Int::Static(a.$checked(*b).expect(OVERFLOW))
                    }
                    _ => Int::from_poly($poly(&self.poly(), &rhs.poly())),
                }
            }
        }
        impl $trait for Int {
            type Output = Int;
            fn $method(self, rhs: Int) -> Int {
                (&self).$method(&rhs)
            }
        }
    };
}

arithmetic!(Add, add, checked_add, |a: &Poly, b: &Poly| a.add(b));
arithmetic!(Sub, sub, checked_sub, |a: &Poly, b: &Poly| a.add(&b.mul(&Poly::constant(-1))));
arithmetic!(Mul, mul, checked_mul, |a: &Poly, b: &Poly| a.mul(b));

impl Neg for &Int {
    type Output = Int;
    fn neg(self) -> Int {
        &Int::Static(0) - self
    }
}

impl From<i64> for Int {
    fn from(value: i64) -> Self {
        Int::Static(value)
    }
}

impl From<Param> for Int {
    fn from(param: Param) -> Self {
        Int::Dynamic(Arc::new(Poly::atom(Atom::Param(param))))
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Int::Static(value) => write!(f, "{value}"),
            Int::Dynamic(poly) => write!(f, "{poly}"),
        }
    }
}

impl fmt::Debug for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
