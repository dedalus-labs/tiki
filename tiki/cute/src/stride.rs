// Copyright © 2026 Dedalus Labs, Inc.

//! Strides: where each mode of a shape steps in the codomain.

use crate::LayoutError;
use crate::offset::Offset;
use crate::shape::Coord;
use crate::tuple::{Profile, Tuple};
use std::fmt;

/// A hierarchical tuple of codomain offsets, all in one codomain.
///
/// The codomain is Z when every stride is an integer, and Z^n when every stride is an
/// arithmetic tuple. Zero belongs to both. [`Stride::try_from`] refuses a mix, so a sum of any
/// strides of one layout is always defined.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Stride(Tuple<Offset>);

impl Stride {
    /// Wraps strides the algebra derived from checked strides.
    pub(crate) fn from_derived(steps: Tuple<Offset>) -> Self {
        Stride(steps)
    }

    pub fn as_tuple(&self) -> &Tuple<Offset> {
        &self.0
    }

    pub fn steps(&self) -> Vec<&Offset> {
        self.0.leaves()
    }

    pub fn modes(&self) -> impl Iterator<Item = Stride> + '_ {
        self.0.modes().iter().cloned().map(Stride)
    }

    /// Returns the offset of a natural coordinate: each coordinate leaf times its stride, summed
    /// (Equation 6).
    pub fn inner_product(&self, coord: &Coord) -> Offset {
        let (coords, steps) = (coord.leaves(), self.steps());
        debug_assert_eq!(coords.len(), steps.len(), "{coord} is a natural coordinate of {self}");
        coords.into_iter().zip(steps).fold(Offset::zero(), |sum, (c, d)| sum.add(&d.scale(c)))
    }

    /// Returns the profile of the codomain: a leaf for Z, one leaf per axis of Z^n.
    pub fn coprofile(&self) -> Profile {
        let sum = self.steps().into_iter().fold(Offset::zero(), |sum, step| sum.add(step));
        sum.as_tuple().profile()
    }
}

/// Checks that the strides share one codomain.
impl TryFrom<Tuple<Offset>> for Stride {
    type Error = LayoutError;

    fn try_from(steps: Tuple<Offset>) -> Result<Self, LayoutError> {
        let leaves = steps.leaves();
        let integers = leaves.iter().any(|d| !d.is_zero() && d.as_int().is_some());
        let tuples = leaves.iter().any(|d| d.as_int().is_none());
        if integers && tuples {
            return Err(LayoutError::MixedCodomain { stride: steps.to_string() });
        }
        Ok(Stride(steps))
    }
}

impl fmt::Display for Stride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for Stride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
