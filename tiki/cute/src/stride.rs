// Copyright © 2026 Dedalus Labs, Inc.

//! Strides: where each mode of a shape steps in the codomain.

use crate::LayoutError;
use crate::offset::Offset;
use crate::shape::Coord;
use crate::tuple::{Profile, Tuple};
use std::fmt;

/// A hierarchical tuple of codomain offsets, all in one codomain.
///
/// The codomain is Z when every stride is an integer, Z^n when every stride is an arithmetic
/// tuple, and the XOR codomain of [`crate::Xor`] when every stride is an XOR value. Zero belongs
/// to all three. [`Stride::try_from`] refuses a mix, so a sum of any strides of one layout is
/// always defined.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Stride(Tuple<Offset>);

impl Stride {
    /// Wraps strides the algebra derived from checked strides.
    pub(crate) fn from_derived(steps: Tuple<Offset>) -> Self {
        Stride(steps)
    }

    /// Returns the strides as a tuple of offsets.
    pub fn as_tuple(&self) -> &Tuple<Offset> {
        &self.0
    }

    /// Returns the stride of every leaf, in leaf order.
    pub fn steps(&self) -> Vec<&Offset> {
        self.0.leaves()
    }

    /// Returns the strides of the top-level modes.
    pub fn modes(&self) -> impl Iterator<Item = Stride> + '_ {
        self.0.modes().iter().cloned().map(Stride)
    }

    /// Returns whether these are XOR strides: some stride is a nonzero XOR value, so every sum
    /// of strides is an XOR.
    pub fn is_xor(&self) -> bool {
        self.steps().into_iter().any(|step| step.as_xor().is_some())
    }

    /// Returns the offset of a natural coordinate: each coordinate leaf times its stride, summed
    /// (Equation 6).
    ///
    /// # Panics
    ///
    /// Panics when these are XOR strides and a coordinate leaf is negative or dynamic, as
    /// [`Offset::scale`] does. [`crate::Layout::call`] checks the coordinate first.
    pub fn inner_product(&self, coord: &Coord) -> Offset {
        let (coords, steps) = (coord.leaves(), self.steps());
        debug_assert_eq!(coords.len(), steps.len(), "{coord} is a natural coordinate of {self}");
        coords.into_iter().zip(steps).fold(Offset::zero(), |sum, (c, d)| sum.add(&d.scale(c)))
    }

    /// Returns the profile of the codomain: a leaf for Z and for the XOR codomain, one leaf per
    /// axis of Z^n.
    ///
    /// The profile is the union of the nonzero strides' own profiles. A sum of the strides would
    /// lose an axis whose strides cancel, such as `2@0` and `-2@0`, although each still steps
    /// along that axis.
    pub fn coprofile(&self) -> Profile {
        let nonzero = self.steps().into_iter().filter(|step| !step.is_zero());
        nonzero.fold(Tuple::Leaf(()), |profile, step| {
            let own = step.as_tuple().map_or(Tuple::Leaf(()), Tuple::profile);
            union(&profile, &own)
        })
    }
}

/// Returns the smallest profile that contains both: a node's modes are unioned position by
/// position, and a leaf stands for an empty profile that any node contains.
fn union(left: &Profile, right: &Profile) -> Profile {
    match (left, right) {
        (Tuple::Node(ours), Tuple::Node(theirs)) => {
            let modes =
                (0..ours.len().max(theirs.len())).map(|i| match (ours.get(i), theirs.get(i)) {
                    (Some(mode), Some(other)) => union(mode, other),
                    (Some(only), None) | (None, Some(only)) => only.clone(),
                    (None, None) => unreachable!("i is below the longer length"),
                });
            Tuple::node(modes)
        }
        (Tuple::Node(_), Tuple::Leaf(())) => left.clone(),
        _ => right.clone(),
    }
}

/// Checks that the strides share one codomain: every pair of strides has a sum. That rules out
/// a nonzero integer beside an arithmetic tuple or an XOR value, and `1@0` beside `1@0@0`, whose
/// first axis is an integer in one and a tuple in the other.
impl TryFrom<Tuple<Offset>> for Stride {
    type Error = LayoutError;

    fn try_from(steps: Tuple<Offset>) -> Result<Self, LayoutError> {
        let summed =
            steps.leaves().into_iter().try_fold(Offset::zero(), |sum, step| sum.checked_add(step));
        if summed.is_none() {
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
