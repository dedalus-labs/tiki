// Copyright © 2026 Dedalus Labs, Inc.

//! Composition `A ∘ B`: the layout that evaluates `A` at every offset of `B` (Cecka, Section
//! 3.3).
//!
//! 1. A tiler of modes composes each mode of `A` with its own tiler.
//! 2. `A` is coalesced along the codomain axes of `B`, so each axis of `A` is one flat
//!    [`Chain`] of `(extent, step)` modes.
//! 3. A hierarchical `B` composes one mode at a time, which distributivity (Equation 19)
//!    allows. The result keeps `B`'s hierarchy.
//! 4. A leaf `B = s:d` walks `A`'s chain once per basis term of `d`: it steps over the whole
//!    modes of `A` that the term's coefficient covers, then takes `s` elements from where it
//!    lands. The admissibility conditions (Equations 20 and 21) are checked on the way. The
//!    walks' results add pointwise.

use crate::LayoutError;
use crate::error::Condition;
use crate::int::Int;
use crate::layout::Layout;
use crate::offset::Offset;
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tiler::Tiler;
use crate::tuple::Tuple;

const OPERATION: &str = "composition";

impl Layout {
    /// Returns `self ∘ tiler`, which maps each coordinate `c` of the tiler to `self(tiler(c))`.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Condition`] when a stride or shape divisibility condition is
    /// violated, or depends on a launch parameter the facts leave open.
    pub fn compose(&self, tiler: &Tiler) -> Result<Layout, LayoutError> {
        match tiler {
            Tiler::Layout(b) => self.compose_layout(b),
            Tiler::Modes(tilers) => self.compose_modes(tilers),
        }
    }

    /// Composes each mode with its tiler. Modes past the last tiler are dropped, because the
    /// result's domain is the tiler's.
    fn compose_modes(&self, tilers: &[Option<Tiler>]) -> Result<Layout, LayoutError> {
        if self.rank() < tilers.len() {
            return Err(LayoutError::TilerRank {
                operation: OPERATION,
                tiler: tilers.len(),
                layout: self.rank(),
            });
        }
        let modes = self.modes().zip(tilers).map(|(mode, tiler)| match tiler {
            Some(tiler) => mode.compose(tiler),
            None => Ok(mode),
        });
        Ok(Layout::from_modes(modes.collect::<Result<Vec<_>, _>>()?))
    }

    fn compose_layout(&self, b: &Layout) -> Result<Layout, LayoutError> {
        // Flatten each codomain axis that B's strides address into one chain, so every walk
        // below sees the leaves of A in the order B steps through them. Coalescing keeps A's
        // function on every integer, so all of B's modes can share this one form of A.
        let a = self.coalesce_z_by(&b.coprofile()).map_err(|error| match error {
            LayoutError::TilerRank { tiler, layout, .. } => {
                LayoutError::TilerRank { operation: OPERATION, tiler, layout }
            }
            other => other,
        })?;

        // Composing B's modes one at a time sums their walks. Check up front that the sum never
        // carries across a mode of A, and note the axes where more than one walk is summed.
        let summed = summed_axes(&a, b)?;
        a.compose_modes_of(b, &summed)
    }

    /// Composes `self`, already coalesced, with each leaf of `b`, keeping `b`'s hierarchy.
    fn compose_modes_of(&self, b: &Layout, summed: &[Vec<usize>]) -> Result<Layout, LayoutError> {
        if b.depth() > 0 {
            let modes = b.modes().map(|mode| self.compose_modes_of(&mode, summed));
            return Ok(Layout::from_modes(modes.collect::<Result<Vec<_>, _>>()?));
        }

        // B is one mode `extent:step`. A zero step revisits one offset, and a single element
        // needs only A's value at the step.
        let extent = b.shape().extents()[0].clone();
        let step = b.stride().steps()[0].clone();
        if step.is_zero() {
            return Ok(leaf_layout(extent, Offset::zero()));
        }
        if extent.is_one() {
            return Ok(leaf_layout(extent, self.call_offset(&step)?));
        }

        // Walk A's chain along each basis term of the step, then add the walks. An integer step
        // is the single term on the empty path, the whole of A.
        let mut sum: Option<Layout> = None;
        for (coefficient, path) in step.terms() {
            let axis = self.get(&path).ok_or_else(|| LayoutError::Mismatch {
                operation: OPERATION,
                detail: format!("{self} has no codomain axis {path:?} for stride {step}"),
            })?;
            let walk = if summed.contains(&path) { Walk::Summed } else { Walk::Sole };
            let term = Chain::from(&axis).walk(coefficient, &extent, walk)?;
            sum = Some(match sum {
                None => term,
                Some(partial) => partial.add(&term)?,
            });
        }
        Ok(sum.expect("a nonzero step has a term").coalesce())
    }
}

/// How a walk's offsets combine with the rest of `B`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Walk {
    /// The only walk along its axis of `A`. It may stop short of the end of a mode that its
    /// stride does not divide, since no other walk adds to its offsets.
    Sole,
    /// One of several walks summed along one axis of `A`. Its stride must split a mode of `A`
    /// exactly (Equation 23, condition 1), or the sum could carry across that mode.
    Summed,
}

/// The flat modes of one codomain axis of `A`, in the order `B` walks them.
struct Chain {
    /// Extent of each remaining mode.
    extents: Vec<Int>,
    /// Codomain step of each remaining mode.
    steps: Vec<Offset>,
}

impl Chain {
    /// Returns `extent` elements of this chain, `stride` elements apart, as a layout.
    fn walk(mut self, stride: Int, extent: &Int, walk: Walk) -> Result<Layout, LayoutError> {
        let stride = self.skip_covered(stride);
        self.enter(&stride, extent, walk)?;
        self.take(extent)
    }

    /// Drops the leading modes that one step of `stride` jumps over entirely, and returns the
    /// stride in units of the first remaining mode.
    ///
    /// A multiple the facts cannot prove stops the skip early. [`Chain::enter`] then checks the
    /// stride against the mode it stopped at, so a layout that needed the skip is refused.
    fn skip_covered(&mut self, mut stride: Int) -> Int {
        while self.extents.len() > 1 && stride.is_multiple_of(&self.extents[0]).is_proven() {
            stride = stride.div_floor(&self.extents[0]);
            self.extents.remove(0);
            self.steps.remove(0);
        }
        stride
    }

    /// Starts the walk inside the first mode: its step becomes `stride` of A's steps, and its
    /// extent shrinks to the `extent[0] / stride` elements the walk visits before it spills into
    /// the next mode (stride divisibility, Equation 20).
    fn enter(&mut self, stride: &Int, extent: &Int, walk: Walk) -> Result<(), LayoutError> {
        self.steps[0] = self.steps[0].scale(stride);
        if self.extents.len() == 1 {
            return Ok(());
        }
        let quotient = self.extents[0].div_floor(stride);
        let divides = self.extents[0].is_multiple_of(stride).and(quotient.is_positive());
        if divides.is_proven() {
            self.extents[0] = quotient;
            return Ok(());
        }
        // Without an exact split a sole walk must stay inside the first mode: its `extent - 1`
        // steps must not pass the mode's end. A summed walk has no such allowance.
        let stays = match walk {
            Walk::Sole => quotient.is_at_least(&(extent - &Int::Static(1))),
            Walk::Summed => divides,
        };
        stays.require(OPERATION, Condition::StrideDivisibility)
    }

    /// Distributes `extent` over the remaining modes, left to right (shape divisibility,
    /// Equation 21). The walk uses each mode in full while what is left exceeds it, and each
    /// full mode must divide what is left. The mode that holds the rest ends the layout.
    fn take(mut self, extent: &Int) -> Result<Layout, LayoutError> {
        let last = self.extents.len() - 1;
        self.extents[last] = extent.clone();
        for i in 0..last {
            // What is left fits inside mode `i`: the walk ends there. This also covers a rest
            // that fills the mode exactly, which PyCuTe splits into the mode and a trailing
            // extent-1 mode that coalescing removes. Deciding "fits" in one test lets a dynamic
            // extent prove it from its facts.
            let left = self.extents[last].clone();
            if self.extents[i].is_at_least(&left).is_proven() {
                self.extents[i] = left;
                self.extents.truncate(i + 1);
                self.steps.truncate(i + 1);
                break;
            }
            left.is_multiple_of(&self.extents[i])
                .require(OPERATION, Condition::ShapeDivisibility)?;
            self.extents[last] = left.div_floor(&self.extents[i]);
        }
        let shape = Shape::from_derived(Tuple::node(self.extents.into_iter().map(Tuple::Leaf)));
        let stride = Stride::from_derived(Tuple::node(self.steps.into_iter().map(Tuple::Leaf)));
        Ok(Layout::from_parts(shape, stride))
    }
}

/// Reads a coalesced axis of `A` as a chain. A coalesced layout is flat, so every mode is a
/// leaf.
impl From<&Layout> for Chain {
    fn from(axis: &Layout) -> Self {
        let extents = axis.shape().extents().into_iter().cloned().collect();
        let steps = axis.stride().steps().into_iter().cloned().collect();
        Chain { extents, steps }
    }
}

/// Returns the codomain axes of `a` along which several walks of `b` are summed, after checking
/// that their images are mutually segregated (Cecka, Equation 23).
///
/// Composing mode by mode evaluates `A(B_0(c_0)) + A(B_1(c_1))` where the definition asks for
/// `A(B_0(c_0) + B_1(c_1))`. The two agree when the sum never carries across a mode of `A`. That
/// holds when each walk splits the modes of `A` exactly and, for every pair of walks on one
/// axis, one walk's whole reach `s * d` ends at or below the other's stride. PyCuTe checks
/// neither, so for `(4, 3):(0, 1)` and `(2, 3):(2, 1)` it returns `(2, 3):(0, 0)` where
/// `A(B(5)) = A(4) = 1`.
///
/// A mode with extent 1 or stride 0 adds nothing to any offset and is left out. A multi-term
/// stride walks once per basis term, each on its own axis. An axis where `a` has one mode is
/// linear, so any sum along it is exact.
fn summed_axes(a: &Layout, b: &Layout) -> Result<Vec<Vec<usize>>, LayoutError> {
    let mut walks: Vec<(Vec<usize>, Int, Int)> = Vec::new();
    for (extent, step) in b.shape().extents().into_iter().zip(b.stride().steps()) {
        if extent.is_one() || step.is_zero() {
            continue;
        }
        for (coefficient, path) in step.terms() {
            if a.get(&path).is_some_and(|axis| axis.rank() > 1) {
                walks.push((path, extent.clone(), coefficient));
            }
        }
    }

    let mut summed: Vec<Vec<usize>> = Vec::new();
    for (i, (path, s_i, d_i)) in walks.iter().enumerate() {
        for (other, s_j, d_j) in &walks[i + 1..] {
            if path != other {
                continue;
            }
            let below = d_j.is_at_least(&(s_i * d_i));
            let above = d_i.is_at_least(&(s_j * d_j));
            below.or(above).require(OPERATION, Condition::SegregatedImages)?;
            if !summed.contains(path) {
                summed.push(path.clone());
            }
        }
    }
    Ok(summed)
}

fn leaf_layout(extent: Int, step: Offset) -> Layout {
    let shape = Shape::from_derived(Tuple::Leaf(extent));
    Layout::from_parts(shape, Stride::from_derived(Tuple::Leaf(step)))
}
