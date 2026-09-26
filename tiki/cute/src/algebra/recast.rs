// Copyright © 2026 Dedalus Labs, Inc.

//! Recasting: the same memory seen through a wider or narrower element.

use crate::LayoutError;
use crate::error::Condition;
use crate::int::Int;
use crate::layout::Layout;
use crate::offset::Offset;
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tuple::Tuple;

const OPERATION: &str = "recast";

impl Layout {
    /// Returns the layout over elements `scale` times as wide, one factor per codomain axis.
    ///
    /// A unit-stride mode shrinks by the factor, rounding up. A mode whose stride is a multiple
    /// of the factor keeps its extent and divides its stride. A factor that is a multiple of the
    /// stride merges that many coordinates into one element.
    ///
    /// # Errors
    ///
    /// Returns [`Condition::RecastDivisibility`] when neither the stride nor the factor divides
    /// the other.
    pub fn recast(&self, scale: &Tuple<Int>) -> Result<Layout, LayoutError> {
        let mut extents = Vec::new();
        let mut steps = Vec::new();
        for (extent, step) in self.shape().extents().into_iter().zip(self.stride().steps()) {
            let (extent, step) = recast_mode(extent, step, scale)?;
            extents.push(extent);
            steps.push(step);
        }
        let shape = Tuple::from_leaves(&mut extents.into_iter(), self.shape().as_tuple());
        let stride = Tuple::from_leaves(&mut steps.into_iter(), self.stride().as_tuple());
        Ok(Layout::from_parts(Shape::from_derived(shape), Stride::from_derived(stride)))
    }
}

/// Recasts one mode `extent:step` by the factor of its codomain axis.
fn recast_mode(
    extent: &Int,
    step: &Offset,
    scale: &Tuple<Int>,
) -> Result<(Int, Offset), LayoutError> {
    // A broadcast mode reads one element whatever its width, so it keeps its extent and needs
    // no factor. Its zero stride also names no codomain axis to look a factor up by.
    if step.is_zero() {
        return Ok((extent.clone(), step.clone()));
    }
    let (d, path) = step
        .as_basis()
        .ok_or_else(|| LayoutError::NotBasis { operation: OPERATION, stride: step.to_string() })?;
    let Some(Tuple::Leaf(factor)) = scale.get(&path) else {
        return Err(LayoutError::Mismatch {
            operation: OPERATION,
            detail: format!("scale {scale} has no factor for codomain axis {path:?}"),
        });
    };
    if !factor.is_positive().is_proven() {
        return Err(LayoutError::Mismatch {
            operation: OPERATION,
            detail: format!("scale factor {factor} for codomain axis {path:?} must be positive"),
        });
    }

    // Recasting keeps a mode's direction, so the arithmetic below runs on the stride's
    // magnitude and puts its sign back. A stride whose sign depends on the launch has no
    // magnitude to divide.
    let direction = if d.is_positive().is_proven() {
        Int::Static(1)
    } else if (-&d).is_positive().is_proven() {
        Int::Static(-1)
    } else {
        return Err(LayoutError::Condition {
            operation: OPERATION,
            condition: Condition::RecastDivisibility,
            verdict: crate::Verdict::Unproven,
        });
    };
    let magnitude = &d * &direction;

    // A unit stride packs `factor` consecutive elements into each wide one. Otherwise the stride
    // or the factor must divide the other, or a wide element would straddle two strided
    // positions: a stride that is a multiple of the factor shrinks by it, and a factor that is a
    // multiple of the stride merges that many coordinates.
    if magnitude.is_one() {
        return Ok((extent.div_ceil(factor), step.clone()));
    }
    let stride_divides = magnitude.is_multiple_of(factor);
    let factor_divides = factor.is_multiple_of(&magnitude);
    stride_divides.or(factor_divides).require(OPERATION, Condition::RecastDivisibility)?;
    if stride_divides.is_proven() {
        let narrowed = &magnitude.div_floor(factor) * &direction;
        return Ok((extent.clone(), Offset::scaled_basis(narrowed, &path)));
    }
    let merged = factor.div_floor(&magnitude);
    Ok((extent.div_ceil(&merged), Offset::scaled_basis(direction, &path)))
}
