// Copyright © 2026 Dedalus Labs, Inc.

//! CuTe XOR transforms with disjoint source and destination fields.
//!
//! 1. Validate the field widths against nonnegative signed 64-bit indices.
//! 2. Copy the source bits onto the destination with exclusive-or (XOR).
//!
//! Applying the same transform twice restores the index. Layout composition
//! supplies the coordinate domain and any offset inside this transformation.
//!
//! A swizzle prints as its constructor, `Swizzle(bits=2, base=0, shift=2)`, and
//! [`Swizzle::cute`] prints CuTe's `SW_2_0_2`, as the `tiki.layout` Python API does.

use crate::LayoutError;
use std::fmt;

/// The three parameters of a swizzle, named so a call site says what each integer means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwizzleParams {
    /// Width of each of the two bit fields.
    pub bits: i64,
    /// Number of untouched low bits below both fields.
    pub base: i64,
    /// Signed distance from the destination field to the source field. A positive shift
    /// copies high bits toward low bits.
    pub shift: i64,
}

/// An immutable index permutation, independent of an array's shape or storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Swizzle {
    /// Width of each disjoint bit field.
    bits: u32,
    /// Number of untouched low bits below both fields.
    base: u32,
    /// Signed distance from destination to source.
    shift: i32,
}

impl Swizzle {
    /// Returns a value that prints this swizzle in CuTe's `SW_bits_base_shift` notation.
    #[must_use]
    pub fn cute(&self) -> impl fmt::Display + '_ {
        CuteSwizzle(self)
    }

    /// Transforms an element offset without accessing memory.
    ///
    /// # Errors
    /// Reject negative indices, which are outside this permutation's domain.
    ///
    /// ```
    /// use tiki_cute::{Swizzle, SwizzleParams};
    /// let transform = Swizzle::try_from(SwizzleParams { bits: 2, base: 0, shift: 2 })?;
    /// assert_eq!(transform.apply(6)?, 7);
    /// assert_eq!(transform.apply(transform.apply(6)?)?, 6);
    /// assert_eq!(transform.to_string(), "Swizzle(bits=2, base=0, shift=2)");
    /// assert_eq!(transform.cute().to_string(), "SW_2_0_2");
    /// # Ok::<(), tiki_cute::LayoutError>(())
    /// ```
    pub fn apply(&self, index: i64) -> Result<i64, LayoutError> {
        if index < 0 {
            return Err(LayoutError::NegativeIndex { index });
        }
        let field = ((1_i64 << self.bits) - 1) << self.base;
        let change = if self.shift >= 0 {
            (index >> self.shift) & field
        } else {
            (index & field) << self.shift.unsigned_abs()
        };
        Ok(index ^ change)
    }

    #[must_use]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    #[must_use]
    pub fn base(&self) -> u32 {
        self.base
    }

    #[must_use]
    pub fn shift(&self) -> i32 {
        self.shift
    }
}

/// Checks a CuTe `Swizzle<bits, base, shift>` for indices in `0..=i64::MAX`.
///
/// The fields must not overlap, `abs(shift) >= bits`, and both must fit in bits 0 through 62,
/// so every result of [`Swizzle::apply`] is a nonnegative `i64`.
impl TryFrom<SwizzleParams> for Swizzle {
    type Error = LayoutError;

    fn try_from(params: SwizzleParams) -> Result<Self, LayoutError> {
        let SwizzleParams { bits, base, shift } = params;
        if !(0..=63).contains(&bits) {
            return Err(LayoutError::Bits { bits });
        }
        if !(0..=63).contains(&base) {
            return Err(LayoutError::Base { base });
        }
        if shift.unsigned_abs() < bits.unsigned_abs() {
            return Err(LayoutError::Overlap { bits, shift });
        }
        if shift.unsigned_abs() > 63 || base + bits + shift.abs() > 63 {
            return Err(LayoutError::Width { bits, base, shift });
        }
        // Every field is now within 0..=63 in magnitude, so the narrowing cannot fail.
        let narrow = "a checked swizzle field fits 32 bits";
        Ok(Swizzle {
            bits: u32::try_from(bits).expect(narrow),
            base: u32::try_from(base).expect(narrow),
            shift: i32::try_from(shift).expect(narrow),
        })
    }
}

impl fmt::Display for Swizzle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Swizzle(bits={}, base={}, shift={})", self.bits, self.base, self.shift)
    }
}

/// CuTe's notation for a swizzle.
struct CuteSwizzle<'a>(&'a Swizzle);

impl fmt::Display for CuteSwizzle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SW_{}_{}_{}", self.0.bits, self.0.base, self.0.shift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swizzle(bits: i64, base: i64, shift: i64) -> Result<Swizzle, LayoutError> {
        Swizzle::try_from(SwizzleParams { bits, base, shift })
    }

    #[test]
    fn invariant_disjoint_fields_are_involutions() {
        for bits in 0..=4 {
            for base in 0..=3 {
                for shift in [-5, 5] {
                    let swizzle = swizzle(bits, base, shift).unwrap();
                    for index in 0..4096 {
                        let mapped = swizzle.apply(index).unwrap();
                        assert_eq!(swizzle.apply(mapped).unwrap(), index);
                        assert_eq!(mapped & ((1 << base) - 1), index & ((1 << base) - 1));
                    }
                }
            }
        }
    }

    #[test]
    fn invariant_invalid_parameters_never_reach_bit_operations() {
        assert!(matches!(swizzle(-1, 0, 2), Err(LayoutError::Bits { .. })));
        assert!(matches!(swizzle(1, -1, 2), Err(LayoutError::Base { .. })));
        for shift in [-1, 0, 1] {
            assert!(matches!(swizzle(2, 0, shift), Err(LayoutError::Overlap { .. })));
        }
        for shift in [i64::MIN, i64::MAX, -64, 64] {
            assert!(matches!(swizzle(1, 0, shift), Err(LayoutError::Width { .. })));
        }
        assert!(matches!(swizzle(1, 62, 1), Err(LayoutError::Width { .. })));
    }

    #[test]
    fn invariant_boundary_indices_remain_representable() {
        for shift in [-1, 1] {
            let swizzle = swizzle(1, 61, shift).unwrap();
            for index in [0, 1, 1 << 61, 1 << 62, i64::MAX] {
                let mapped = swizzle.apply(index).unwrap();
                assert!(mapped >= 0);
                assert_eq!(swizzle.apply(mapped).unwrap(), index);
            }
            assert!(matches!(swizzle.apply(-1), Err(LayoutError::NegativeIndex { .. })));
        }
        let identity = swizzle(0, 63, 0).unwrap();
        assert_eq!(identity.apply(i64::MAX).unwrap(), i64::MAX);
    }
}
