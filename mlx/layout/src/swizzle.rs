// Copyright © 2026 Dedalus Labs, Inc.

//! CuTe XOR transforms with disjoint source and destination fields.
//!
//! 1. Validate the field widths against nonnegative signed 64-bit indices.
//! 2. Copy the source bits onto the destination with exclusive-or (XOR).
//!
//! Applying the same transform twice restores the index. Layout composition
//! supplies the coordinate domain and any offset inside this transformation.

use crate::LayoutError;

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
    /// Validate a CuTe `Swizzle<bits, base, shift>` for indices in `0..=i64::MAX`.
    ///
    /// # Errors
    /// Reject overlapping fields or parameters outside the index representation.
    pub fn new(bits: i64, base: i64, shift: i64) -> Result<Self, LayoutError> {
        if !(0..=63).contains(&bits) {
            return Err(LayoutError::Bits { bits });
        }
        if !(0..=63).contains(&base) {
            return Err(LayoutError::Base { base });
        }
        if shift.unsigned_abs() < bits as u64 {
            return Err(LayoutError::Overlap { bits, shift });
        }
        if shift.unsigned_abs() > 63 || base + bits + shift.abs() > 63 {
            return Err(LayoutError::Width { bits, base, shift });
        }
        Ok(Self { bits: bits as u32, base: base as u32, shift: shift as i32 })
    }

    /// Transform an element offset without accessing memory.
    ///
    /// # Errors
    /// Reject negative indices, which are outside this permutation's domain.
    ///
    /// ```
    /// use tiki_layout::Swizzle;
    /// let transform = Swizzle::new(2, 0, 2)?;
    /// assert_eq!(transform.apply(6)?, 7);
    /// assert_eq!(transform.apply(transform.apply(6)?)?, 6);
    /// # Ok::<(), tiki_layout::LayoutError>(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariant_disjoint_fields_are_involutions() {
        for bits in 0..=4 {
            for base in 0..=3 {
                for shift in [-5, 5] {
                    let swizzle = Swizzle::new(bits, base, shift).unwrap();
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
        assert!(matches!(Swizzle::new(-1, 0, 2), Err(LayoutError::Bits { .. })));
        assert!(matches!(Swizzle::new(1, -1, 2), Err(LayoutError::Base { .. })));
        for shift in [-1, 0, 1] {
            assert!(matches!(Swizzle::new(2, 0, shift), Err(LayoutError::Overlap { .. })));
        }
        for shift in [i64::MIN, i64::MAX, -64, 64] {
            assert!(matches!(Swizzle::new(1, 0, shift), Err(LayoutError::Width { .. })));
        }
        assert!(matches!(Swizzle::new(1, 62, 1), Err(LayoutError::Width { .. })));
    }

    #[test]
    fn invariant_boundary_indices_remain_representable() {
        for shift in [-1, 1] {
            let swizzle = Swizzle::new(1, 61, shift).unwrap();
            for index in [0, 1, 1 << 61, 1 << 62, i64::MAX] {
                let mapped = swizzle.apply(index).unwrap();
                assert!(mapped >= 0);
                assert_eq!(swizzle.apply(mapped).unwrap(), index);
            }
            assert!(matches!(swizzle.apply(-1), Err(LayoutError::NegativeIndex { .. })));
        }
        let identity = Swizzle::new(0, 63, 0).unwrap();
        assert_eq!(identity.apply(i64::MAX).unwrap(), i64::MAX);
    }
}
