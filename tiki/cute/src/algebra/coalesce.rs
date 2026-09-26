// Copyright © 2026 Dedalus Labs, Inc.

//! Coalescing: the flattest layout with the same function (Cecka, Equations 13 to 15).
//!
//! Two adjacent leaf modes `s_a:d_a` and `s_b:d_b` act as one mode `s_a*s_b:d_a` exactly when
//! `d_b = s_a*d_a`: stepping off the end of the first mode lands where the second one starts.
//! Coalescing applies that rule left to right and drops extent-1 modes, which add nothing to
//! any offset.
//!
//! An XOR mode also needs `s_a` to be a power of two. With every product carry-less, the pair
//! maps index `a + s_a*b`, for `a < s_a`, to `(a ^ s_a*b) * d_a`, and the merged mode maps it to
//! `(a + s_a*b) * d_a`. The two agree for every `a` and `b` exactly when `s_a` is a power of two,
//! which is what PyCuTe's second linearity check, at `(s_a - 1, 1)`, decides.

use crate::LayoutError;
use crate::int::Int;
use crate::layout::Layout;
use crate::offset::Offset;
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tuple::{Profile, Tuple};

/// One mode of a flat layout: its extent and its step.
pub(crate) type FlatMode = (Int, Offset);

impl Layout {
    /// Returns the flat layout with the same function on every integer, past the end included.
    ///
    /// A trailing extent-1 mode keeps its stride, because an index past the end lands in it.
    /// Composition needs that property: it evaluates the left layout at offsets beyond its size.
    #[must_use]
    pub fn coalesce_z(&self) -> Layout {
        from_flat(flat_modes(self))
    }

    /// Returns the flat layout with the same function on `0..size`.
    #[must_use]
    pub fn coalesce(&self) -> Layout {
        let mut modes = flat_modes(self);
        // Inside the domain a trailing extent-1 mode only sees coordinate 0. A lone mode stays,
        // because a layout has at least one.
        if modes.len() > 1 && modes.last().is_some_and(|(extent, _)| extent.is_one()) {
            modes.pop();
        }
        from_flat(modes)
    }

    /// Applies [`Layout::coalesce_z`] to each part `profile` names. A profile leaf coalesces the
    /// whole mode opposite it, and modes past the profile's rank stay as they are.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::TilerRank`] when the profile has more modes than the layout.
    pub fn coalesce_z_by(&self, profile: &Profile) -> Result<Layout, LayoutError> {
        self.coalesce_modes(profile, &Layout::coalesce_z)
    }

    /// Applies [`Layout::coalesce`] to each part `profile` names, as [`Layout::coalesce_z_by`].
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::TilerRank`] when the profile has more modes than the layout.
    pub fn coalesce_by(&self, profile: &Profile) -> Result<Layout, LayoutError> {
        self.coalesce_modes(profile, &Layout::coalesce)
    }

    fn coalesce_modes(
        &self,
        profile: &Profile,
        coalesce: &impl Fn(&Layout) -> Layout,
    ) -> Result<Layout, LayoutError> {
        let Tuple::Node(parts) = profile else { return Ok(coalesce(self)) };
        if self.rank() < parts.len() {
            return Err(LayoutError::TilerRank {
                operation: "coalesce",
                tiler: parts.len(),
                layout: self.rank(),
            });
        }
        let modes = self.modes().enumerate().map(|(i, mode)| match parts.get(i) {
            Some(part) => mode.coalesce_modes(part, coalesce),
            None => Ok(mode),
        });
        Ok(Layout::from_modes(modes.collect::<Result<Vec<_>, _>>()?))
    }
}

/// Returns the leaf modes of `layout`, merged left to right.
///
/// Both tests are structural. Canonical polynomials that are equal are equal for every launch,
/// so a merge is always sound. A merge that holds for only some launches is skipped, which
/// leaves a correct layout that is less coalesced.
pub(crate) fn flat_modes(layout: &Layout) -> Vec<FlatMode> {
    // An XOR mode merges only across a power-of-two extent, and never across a dynamic one.
    let joins = |extent: &Int, step: &Offset| {
        step.as_xor().is_none() || extent.as_static().is_some_and(|s| s.count_ones() == 1)
    };
    let mut merged: Vec<FlatMode> = Vec::new();
    for (extent, step) in layout.shape().extents().into_iter().zip(layout.stride().steps()) {
        // An extent-1 mode before this one contributes nothing and must not block a merge.
        while merged.last().is_some_and(|(extent, _)| extent.is_one()) {
            merged.pop();
        }
        if let Some((last_extent, last_step)) = merged.last_mut()
            && last_step.checked_scale(last_extent).as_ref() == Some(step)
            && joins(last_extent, last_step)
        {
            *last_extent = &*last_extent * extent;
            continue;
        }
        merged.push((extent.clone(), step.clone()));
    }
    merged
}

/// Returns the flat layout of `modes`: a leaf layout for one mode, and `1:0` for none.
pub(crate) fn from_flat(modes: Vec<FlatMode>) -> Layout {
    if modes.is_empty() {
        let shape = Shape::from_derived(Tuple::Leaf(Int::Static(1)));
        return Layout::from_parts(shape, Stride::from_derived(Tuple::Leaf(Offset::zero())));
    }
    let (extents, steps): (Vec<_>, Vec<_>) =
        modes.into_iter().map(|(s, d)| (Tuple::Leaf(s), Tuple::Leaf(d))).unzip();
    let shape = Shape::from_derived(Tuple::Node(extents).unwrap());
    Layout::from_parts(shape, Stride::from_derived(Tuple::Node(steps).unwrap()))
}
