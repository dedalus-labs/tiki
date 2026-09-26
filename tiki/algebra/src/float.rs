// Copyright © 2026 Dedalus Labs, Inc.

//! Floats under the real, tropical and log semirings.
//!
//! Floating-point addition is associative only up to rounding, so these implementations hold
//! their laws up to rounding. Regrouping a fold changes its rounding, never its meaning.

use crate::operator::{Add, LogSumExp, Max, Min, Mul};
use crate::structure::{CommutativeMonoid, Magma, Monoid, Semigroup, Semiring};

/// Implements the float monoids for one float type.
macro_rules! float_structures {
    ($float:ty) => {
        impl Magma<Add> for $float {
            fn operate(&self, other: &Self) -> Self {
                self + other
            }
        }

        impl Magma<Mul> for $float {
            fn operate(&self, other: &Self) -> Self {
                self * other
            }
        }

        impl Magma<Max> for $float {
            fn operate(&self, other: &Self) -> Self {
                self.max(*other)
            }
        }

        impl Magma<Min> for $float {
            fn operate(&self, other: &Self) -> Self {
                self.min(*other)
            }
        }

        /// `log(eᵃ + eᵇ)`, computed as `max + log1p(e^-|a - b|)` so no exponential overflows.
        impl Magma<LogSumExp> for $float {
            fn operate(&self, other: &Self) -> Self {
                let (high, low) = if self >= other { (*self, *other) } else { (*other, *self) };
                if low == <$float>::NEG_INFINITY {
                    return high;
                }
                high + (low - high).exp().ln_1p()
            }
        }

        impl Semigroup<Add> for $float {}
        impl Semigroup<Mul> for $float {}
        impl Semigroup<Max> for $float {}
        impl Semigroup<Min> for $float {}
        impl Semigroup<LogSumExp> for $float {}

        impl Monoid<Add> for $float {
            fn identity() -> Self {
                0.0
            }
        }

        impl Monoid<Mul> for $float {
            fn identity() -> Self {
                1.0
            }
        }

        impl Monoid<Max> for $float {
            fn identity() -> Self {
                <$float>::NEG_INFINITY
            }
        }

        impl Monoid<Min> for $float {
            fn identity() -> Self {
                <$float>::INFINITY
            }
        }

        impl Monoid<LogSumExp> for $float {
            fn identity() -> Self {
                <$float>::NEG_INFINITY
            }
        }

        impl CommutativeMonoid<Add> for $float {}
        impl CommutativeMonoid<Mul> for $float {}
        impl CommutativeMonoid<Max> for $float {}
        impl CommutativeMonoid<Min> for $float {}
        impl CommutativeMonoid<LogSumExp> for $float {}

        /// The real semiring, `(+, ×)`.
        impl Semiring<Add, Mul> for $float {}
        /// The tropical semiring, `(max, +)`.
        impl Semiring<Max, Add> for $float {}
        /// The min-plus semiring, `(min, +)`, for shortest paths.
        impl Semiring<Min, Add> for $float {}
        /// The log semiring, `(logsumexp, +)`.
        impl Semiring<LogSumExp, Add> for $float {}
    };
}

float_structures!(f32);
float_structures!(f64);
