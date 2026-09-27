// Copyright © 2026 Dedalus Labs, Inc.

//! Complement: the modes that fill the gaps of an injective layout's image (Cecka, Equations 27
//! to 29).
//!
//! Sorted by stride, the modes of `A` expose the gaps. A mode `s:d` above a chain that ends at
//! offset `e` leaves `d / e` uncovered blocks of size `e` below it. The complement enumerates
//! those blocks, one mode per gap, and a last mode that repeats the whole pattern. Each codomain
//! axis of `A` is complemented separately.

use crate::LayoutError;
use crate::error::Condition;
use crate::int::Int;
use crate::layout::Layout;
use crate::offset::{Offset, static_first};
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tiler::Tiler;
use crate::tuple::{Profile, Tuple};

const OPERATION: &str = "complement";

impl Layout {
    /// Returns the ordered layout whose offsets, added to this layout's, tile the codomain.
    ///
    /// # Errors
    ///
    /// Returns [`Condition::OrderedChain`] when a mode starts before the end of the chain below
    /// it, which covers overlapping and interleaved modes, or when the parameter facts cannot
    /// place a dynamic stride past that end.
    pub fn complement(&self) -> Result<Layout, LayoutError> {
        // Every codomain axis starts with no gaps and a covered prefix that ends at offset 1.
        let profile = self.coprofile();
        let mut gaps: Vec<Gaps> = profile.leaf_paths().into_iter().map(Gaps::new).collect();

        // Visit the modes in stride order, so each one starts at or past the end of the chain
        // below it. Static strides sort by value, and dynamic ones follow in their own order.
        // `Gaps::cover` proves every placement, so an order the facts cannot confirm refuses.
        let mut modes: Vec<(&Offset, &Int)> =
            self.stride().steps().into_iter().zip(self.shape().extents()).collect();
        modes.sort_by(|(a, _), (b, _)| static_first(a, b));
        for (step, extent) in modes {
            let (d, path) = step.as_basis().ok_or_else(|| LayoutError::NotBasis {
                operation: OPERATION,
                stride: step.to_string(),
            })?;
            if d.is_zero() || extent.is_one() {
                continue;
            }
            let chain = gaps.iter_mut().find(|chain| chain.path == path).expect("a coprofile path");
            chain.cover(&d, extent)?;
        }

        // Close each axis and place the axes back into the codomain, each on its own basis.
        let layouts: Vec<Layout> = gaps.into_iter().map(Gaps::close).collect();
        let per_axis = Tuple::from_leaves(&mut layouts.into_iter(), &profile);
        Tiler::from(per_axis).to_layout()
    }

    /// Returns the complement grown to cover `cotarget`, as CuTe's `complement(layout,
    /// cotarget)`. The last mode of each axis extends over the part of `cotarget` the
    /// complement has not reached, so the result enumerates all of `cotarget`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Layout::complement`], and [`LayoutError::Mismatch`] when
    /// `cotarget` has fewer modes than the codomain has axes.
    pub fn complement_in(&self, cotarget: &Shape) -> Result<Layout, LayoutError> {
        let complement = self.complement()?;
        extend(&self.coprofile(), &complement, cotarget, &mut Vec::new())
    }
}

/// The complement under construction along one codomain axis.
struct Gaps {
    /// Where this axis sits in the codomain, as a basis path.
    path: Vec<usize>,
    /// Extent of each gap found so far, in blocks of the chain below it.
    extents: Vec<Int>,
    /// Stride of each gap, followed by the offset where the covered chain ends. It holds one
    /// more entry than `extents`, and its last entry is the next gap's stride.
    strides: Vec<Int>,
}

impl Gaps {
    fn new(path: Vec<usize>) -> Gaps {
        Gaps { path, extents: Vec::new(), strides: vec![Int::Static(1)] }
    }

    /// Records a mode of `extent` elements `stride` apart as the next part of the chain.
    fn cover(&mut self, stride: &Int, extent: &Int) -> Result<(), LayoutError> {
        let end = self.strides.last().expect("the chain has an end");
        stride.is_at_least(end).require(OPERATION, Condition::OrderedChain)?;
        self.extents.push(stride.div_floor(end));
        self.strides.push(stride * extent);
        Ok(())
    }

    /// Returns the complement along this axis: the gaps, then an extent-1 mode at the chain's
    /// end that [`Layout::complement_in`] grows.
    fn close(self) -> Layout {
        let extents = self.extents.into_iter().chain([Int::Static(1)]).map(Tuple::Leaf);
        let strides = self.strides.into_iter().map(|d| Tuple::Leaf(Offset::from(d)));
        let shape = Shape::from_derived(Tuple::node(extents));
        Layout::from_parts(shape, Stride::from_derived(Tuple::node(strides))).coalesce_z()
    }
}

/// Grows the complement along each axis of `profile` into the matching part of `cotarget`.
/// Modes of `cotarget` past the codomain's axes are new axes, each enumerated on its own basis.
fn extend(
    profile: &Profile,
    complement: &Layout,
    cotarget: &Shape,
    path: &mut Vec<usize>,
) -> Result<Layout, LayoutError> {
    let Tuple::Node(profiles) = profile else { return Ok(grow(complement, cotarget)) };
    if cotarget.rank() < profiles.len() {
        return Err(LayoutError::Mismatch {
            operation: OPERATION,
            detail: format!(
                "cotarget {cotarget} has fewer modes than the {} codomain axes",
                profiles.len()
            ),
        });
    }
    let mut modes = Vec::with_capacity(cotarget.rank());
    for (i, target) in cotarget.modes().enumerate() {
        path.push(i);
        let mode = match (profiles.get(i), complement.get(&[i])) {
            (Some(part_profile), Some(part)) => extend(part_profile, &part, &target, path)?,
            _ => Layout::from_parts(target.clone(), basis_like(&target, path)),
        };
        modes.push(mode);
        path.pop();
    }
    Ok(Layout::from_modes(modes))
}

/// Replaces the complement's final extent-1 mode with the part of `cotarget` past its reach.
fn grow(complement: &Layout, cotarget: &Shape) -> Layout {
    // The last stride is where the complement's reach ends. Counted in multiples of that reach,
    // each cotarget mode still needs `ceil(extent / reach)` steps, and each mode it covers
    // shrinks the reach left for the next.
    let steps = complement.stride().steps();
    let last = steps.last().expect("a complement has a mode");
    let (mut reach, _) = last.as_basis().expect("a complement axis has basis strides");
    let mut rest = Vec::new();
    for extent in cotarget.extents() {
        rest.push(Tuple::Leaf(extent.div_ceil(&reach)));
        reach = reach.div_ceil(extent);
    }

    // The final stride leaf expands over the new modes as their compact strides.
    let mut extents: Vec<Tuple<Int>> = complement.shape().as_tuple().modes().to_vec();
    extents.pop();
    extents.push(Tuple::Node(rest));
    let strides = Tuple::node(complement.stride().as_tuple().modes().iter().cloned());
    Layout::from_expanded(Shape::from_derived(Tuple::Node(extents)), &strides).coalesce()
}

/// Returns strides `E(path ++ p)` for each leaf path `p` of `shape`, so each leaf enumerates
/// its own codomain axis.
fn basis_like(shape: &Shape, path: &[usize]) -> Stride {
    let paths = shape.as_tuple().leaf_paths();
    let steps = paths.into_iter().map(|leaf| {
        let full: Vec<usize> = path.iter().chain(&leaf).copied().collect();
        Offset::basis(&full)
    });
    Stride::from_derived(Tuple::from_leaves(
        &mut steps.collect::<Vec<_>>().into_iter(),
        shape.as_tuple(),
    ))
}
