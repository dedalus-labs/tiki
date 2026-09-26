// Copyright © 2026 Dedalus Labs, Inc.

//! Inverses and the null space (Cecka, Equations 24 to 26).
//!
//! A layout is rarely a bijection, so its inverses are partial. The right inverse `R` inverts
//! the contiguous part of the image: `A(R(k)) = k` for every `k` in `0..size(R)`. The left
//! inverse `L` recovers a coordinate from anywhere in the image: `A(L(A(i))) = A(i)`. Both walk
//! the coalesced modes of `A` in stride order, one codomain axis at a time.

use crate::LayoutError;
use crate::algebra::coalesce::{flat_modes, from_flat};
use crate::error::Condition;
use crate::int::Int;
use crate::layout::Layout;
use crate::offset::{Offset, static_first};
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tuple::Tuple;
use std::cmp::Ordering;

const LEFT: &str = "left inverse";

/// A coalesced mode of `A` and the domain index where it starts.
struct Mode {
    /// Codomain step of the mode.
    step: Offset,
    /// Number of coordinates in the mode.
    extent: Int,
    /// Domain index of the mode's first step, the product of the extents before it. The
    /// inverse maps the mode's offsets back to multiples of this.
    index: Int,
}

impl Layout {
    /// Returns the layout `R` with `self(R(k)) = k` for every `k` in `0..size(R)`: the longest
    /// chain of modes of `self` that covers the start of each codomain axis without gaps.
    #[must_use]
    pub fn right_inverse(&self) -> Layout {
        let modes = Mode::sorted(self);
        let profile = self.coprofile();
        let per_axis = profile.leaf_paths().into_iter().map(|path| {
            // From offset `E(axis)`, repeatedly take the mode whose step is exactly where the
            // covered prefix of the axis ends. A mode that starts anywhere else would leave a
            // gap or an overlap, so it cannot extend the inverse.
            let mut end = Offset::basis(&path);
            let mut inverse = Vec::new();
            for mode in &modes {
                if mode.extent.is_one() || mode.step != end {
                    continue;
                }
                inverse.push((mode.extent.clone(), Offset::from(mode.index.clone())));
                end = end.scale(&mode.extent);
            }
            from_flat(inverse).coalesce()
        });
        Layout::from(Tuple::from_leaves(&mut per_axis.collect::<Vec<_>>().into_iter(), &profile))
    }

    /// Returns a layout `L` with `self(L(self(i))) = self(i)` for every `i` in `0..size`.
    ///
    /// # Errors
    ///
    /// Returns [`Condition::OrderedChain`] when a stride is not a multiple of the extents below
    /// it, and [`Condition::Injective`] when two modes overlap.
    pub fn left_inverse(&self) -> Result<Layout, LayoutError> {
        let profile = self.coprofile();
        let paths = profile.leaf_paths();
        let mut chains: Vec<Chain> = paths.iter().map(|_| Chain::new()).collect();

        // Equal strides break ties by extent, then index, as PyCuTe's tuple sort does, so both
        // report the same pair of overlapping modes.
        let mut modes = Mode::sorted(self);
        modes.sort_by(|a, b| {
            static_first(&a.step, &b.step)
                .then_with(|| a.extent.compare(&b.extent).unwrap_or(Ordering::Equal))
                .then_with(|| a.index.compare(&b.index).unwrap_or(Ordering::Equal))
        });
        for mode in modes {
            let (d, path) = mode.step.as_basis().ok_or_else(|| LayoutError::NotBasis {
                operation: LEFT,
                stride: mode.step.to_string(),
            })?;
            if d.is_zero() || mode.extent.is_one() {
                continue;
            }
            let axis = paths.iter().position(|p| *p == path).expect("a coprofile path");
            chains[axis].extend(&d, mode)?;
        }

        let per_axis: Vec<Layout> = chains.into_iter().map(Chain::into_layout).collect();
        Ok(Layout::from(Tuple::from_leaves(&mut per_axis.into_iter(), &profile)))
    }

    /// Returns the layout of the domain indices this layout sends to 0: one mode per stride-0
    /// leaf, stepping by that leaf's position in the domain.
    #[must_use]
    pub fn nullspace(&self) -> Layout {
        let index = self.shape().compact_stride(&Offset::from(1));
        let leaves =
            self.shape().extents().into_iter().zip(self.stride().steps()).zip(index.steps());
        let zeros = leaves.filter(|((_, step), _)| step.is_zero());
        from_flat(zeros.map(|((extent, _), index)| (extent.clone(), index.clone())).collect())
    }
}

impl Mode {
    /// Returns the coalesced modes of `layout`, stably sorted by step. Dynamic steps follow the
    /// static ones in their own order, and each caller proves what it assumes of that order.
    fn sorted(layout: &Layout) -> Vec<Mode> {
        let mut index = Int::Static(1);
        let mut modes: Vec<Mode> = flat_modes(layout)
            .into_iter()
            .map(|(extent, step)| {
                let start = index.clone();
                index = &index * &extent;
                Mode { step, extent, index: start }
            })
            .collect();
        modes.sort_by(|a, b| static_first(&a.step, &b.step));
        modes
    }
}

/// The left inverse under construction along one codomain axis.
struct Chain {
    /// Extent of each inverse mode. The first is a placeholder that the first gap overwrites,
    /// so offsets before the first mode map to index 0.
    extents: Vec<Int>,
    /// Domain index each inverse mode steps by.
    indices: Vec<Int>,
    /// The codomain distance the chain covers so far.
    covered: Int,
}

impl Chain {
    fn new() -> Chain {
        Chain {
            extents: vec![Int::Static(1)],
            indices: vec![Int::Static(0)],
            covered: Int::Static(1),
        }
    }

    /// Appends a mode of `A` whose step is `stride` along this axis.
    fn extend(&mut self, stride: &Int, mode: Mode) -> Result<(), LayoutError> {
        // The gap from the covered chain to this mode becomes the extent of the previous inverse
        // mode. It must be a whole multiple of the covered distance, and at least the previous
        // mode's own extent, or two coordinates would share an offset.
        stride.is_multiple_of(&self.covered).require(LEFT, Condition::OrderedChain)?;
        let gap = stride.div_floor(&self.covered);
        let previous = self.extents.last().expect("a chain starts with one mode");
        gap.is_at_least(previous).require(LEFT, Condition::Injective)?;

        *self.extents.last_mut().expect("a chain starts with one mode") = gap.clone();
        self.covered = &self.covered * &gap;
        self.extents.push(mode.extent);
        self.indices.push(mode.index);
        Ok(())
    }

    fn into_layout(self) -> Layout {
        let shape = Shape::from_derived(Tuple::node(self.extents.into_iter().map(Tuple::Leaf)));
        let steps = self.indices.into_iter().map(|index| Tuple::Leaf(Offset::from(index)));
        Layout::from_parts(shape, Stride::from_derived(Tuple::node(steps))).coalesce_z()
    }
}
