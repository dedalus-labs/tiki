// Copyright © 2026 Dedalus Labs, Inc.

//! XOR strides: the codomain whose addition is XOR, defined on [`Xor`].

use crate::LayoutError;
use crate::int::Int;
use crate::poly::OVERFLOW;
use std::fmt;
use std::ops::{Add, Mul};

/// A value of the XOR codomain, held as the bits of a nonnegative `i64`.
///
/// F2 is the field {0, 1}, whose addition is XOR. An XOR value is a vector over F2, read as the
/// bits of a nonnegative integer and written `^v`, so `^5 + ^3` is `^6`. An integer acts on it
/// by carry-less multiplication: `n * ^v` is the XOR of `v` shifted left by each set bit of `n`,
/// so `3 * ^3` is `^5` where `3 * 3` is 9 in Z. The two products agree exactly when the
/// schoolbook product carries no bit, as it never does for a power of two.
///
/// XOR strides are strides whose values are XOR values. A layout with XOR strides maps a
/// coordinate to the XOR of its carry-less products with the strides, which makes a swizzle an
/// ordinary layout: `(8, 8):(^1, ^9)` maps `(r, c)` to `r ^ 9c`, which is [`crate::Swizzle`]
/// `(bits=3, base=0, shift=3)` applied to `r + 8c`. PyCuTe calls these strides `F2`.
///
/// ```
/// use tiki_cute::{Layout, Offset, Xor};
///
/// let swizzled: Layout = "(8, 8):(^1, ^9)".parse()?;
/// assert_eq!(swizzled.at(2 + 8 * 3), Offset::from(Xor::try_from(2 ^ 27)?));
/// # Ok::<(), tiki_cute::LayoutError>(())
/// ```
///
/// `+` is XOR and `*` is the carry-less product. Values order by their integer value, so a sort
/// by stride visits XOR strides by leading bit, as PyCuTe's sorts do. The carry-less product
/// panics when its result needs more than 63 bits, as [`Int`] arithmetic panics on overflow.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Xor(i64);

impl Xor {
    /// The zero vector, the additive identity.
    pub(crate) const ZERO: Xor = Xor(0);
    /// The carry-less multiplicative identity.
    pub(crate) const ONE: Xor = Xor(1);

    /// Returns the bits as a nonnegative integer.
    #[must_use]
    pub fn value(self) -> i64 {
        self.0
    }

    /// Returns whether this is the zero vector.
    pub(crate) fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Returns `(quotient, remainder)` of polynomial division over F2, the unique pair with
    /// `self = quotient * divisor + remainder` and fewer bits in the remainder than in the
    /// divisor. A power-of-two divisor splits the bits: `^22` by 4 is `(^5, ^2)`.
    ///
    /// # Panics
    ///
    /// Panics when `divisor` is zero. Every divisor the algebra passes is a positive extent.
    pub(crate) fn div_rem(self, divisor: Xor) -> (Xor, Xor) {
        assert!(!divisor.is_zero(), "division of {self} by zero");
        let degree = bit_length(divisor.0);
        let (mut quotient, mut remainder) = (0, self.0);
        // Cancel the remainder's leading bit until it has fewer bits than the divisor.
        while bit_length(remainder) >= degree {
            let shift = bit_length(remainder) - degree;
            quotient ^= 1 << shift;
            remainder ^= divisor.0 << shift;
        }
        (Xor(quotient), Xor(remainder))
    }
}

/// Returns the number of bits up to and including the leading one, 0 for 0.
pub(crate) fn bit_length(value: i64) -> u32 {
    i64::BITS - value.leading_zeros()
}

/// XOR, the addition of F2 applied bitwise.
impl Add for Xor {
    type Output = Xor;

    #[allow(clippy::suspicious_arithmetic_impl)] // The addition of F2 is XOR.
    fn add(self, other: Xor) -> Xor {
        Xor(self.0 ^ other.0)
    }
}

/// The carry-less product: the XOR of `self` shifted by each set bit of `other`.
impl Mul for Xor {
    type Output = Xor;

    fn mul(self, other: Xor) -> Xor {
        if self.is_zero() || other.is_zero() {
            return Xor::ZERO;
        }
        // The product's leading bit is the sum of the leading bits, so its width is known first.
        assert!(bit_length(self.0) + bit_length(other.0) <= i64::BITS, "{OVERFLOW}");
        let mut product = 0;
        let mut rest = other.0;
        let mut shifted = self.0;
        while rest != 0 {
            if rest & 1 == 1 {
                product ^= shifted;
            }
            rest >>= 1;
            if rest != 0 {
                shifted <<= 1;
            }
        }
        Xor(product)
    }
}

/// Reads a nonnegative integer as its bits.
impl TryFrom<i64> for Xor {
    type Error = LayoutError;

    fn try_from(value: i64) -> Result<Self, LayoutError> {
        if value < 0 {
            return Err(LayoutError::XorOperand { operand: value.to_string() });
        }
        Ok(Xor(value))
    }
}

/// Reads a static nonnegative integer as its bits. A dynamic integer has no bits the compiler
/// knows, so it has no XOR value.
impl TryFrom<&Int> for Xor {
    type Error = LayoutError;

    fn try_from(value: &Int) -> Result<Self, LayoutError> {
        let operand = || LayoutError::XorOperand { operand: value.to_string() };
        Xor::try_from(value.as_static().ok_or_else(operand)?).map_err(|_| operand())
    }
}

/// Prints `^v`, the notation [`crate::Layout`] parses back.
impl fmt::Display for Xor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "^{}", self.0)
    }
}

impl fmt::Debug for Xor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xor(value: i64) -> Xor {
        Xor::try_from(value).unwrap()
    }

    #[test]
    fn invariant_product_is_carry_less() {
        assert_eq!(xor(5) * xor(3), xor(15));
        assert_eq!(xor(3) * xor(3), xor(0b101));
        assert_eq!(xor(0b1010) * xor(2), xor(0b10100));
        assert_eq!(xor(7) * Xor::ZERO, Xor::ZERO);
        assert_eq!(xor(1 << 62) * Xor::ONE, xor(1 << 62));
    }

    #[test]
    fn invariant_division_inverts_the_product() {
        for a in 0..64 {
            for b in 1..20 {
                let (quotient, remainder) = xor(a).div_rem(xor(b));
                assert_eq!(quotient * xor(b) + remainder, xor(a));
                assert!(bit_length(remainder.0) < bit_length(b));
            }
        }
        assert_eq!(xor(0b1011).div_rem(xor(0b11)), (xor(0b110), xor(1)));
        assert_eq!(xor(0b10110).div_rem(xor(4)), (xor(0b101), xor(0b10)));
    }

    #[test]
    #[should_panic(expected = "overflows")]
    fn invariant_wide_products_panic() {
        let _ = xor(1 << 62) * xor(2);
    }

    #[test]
    fn invariant_negative_integers_have_no_bits() {
        assert!(matches!(Xor::try_from(-1), Err(LayoutError::XorOperand { .. })));
    }
}
