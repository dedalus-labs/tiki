// Copyright © 2026 Dedalus Labs, Inc.

//! Tensors: a layout placed at an element offset inside bounded storage.
//!
//! Cecka defines a tensor as an accessor composed with a layout (Section 2.4). Tiki's accessor is
//! an element offset into storage whose byte range is known, so every tensor carries the proof
//! that each of its elements lies inside that range. [`Tensor::new`] checks it, the fields are
//! private, and every derived tensor passes through `new` again. A kernel handed a `Tensor`
//! therefore cannot address a byte outside the storage it came from. Tensors never own or read
//! storage.
//!
//! The byte range is computed exactly, from the least and greatest offset of the layout, so
//! negative strides and broadcast modes are covered. A tensor's layout must be static and map
//! into Z. Dynamic layouts reach memory only through the kernel compiler, which proves the same
//! bound symbolically.

use crate::LayoutError;
use crate::int::Int;
use crate::layout::Layout;
use crate::tiler::Tiler;
use crate::tuple::Tuple;

/// The half-open byte range `[start, end)` of storage a tensor may read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    /// First readable byte.
    pub start: u64,
    /// One past the last readable byte.
    pub end: u64,
}

/// A static layout placed at `offset` elements into storage, with every element inside
/// `bounds`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    /// Maps each coordinate to an element offset from `offset`.
    layout: Layout,
    /// Element offset of coordinate 0 from the start of storage.
    offset: i64,
    /// Bytes per element, never zero.
    element_size: u64,
    /// The storage every element of the tensor lies in.
    bounds: Bounds,
}

impl Tensor {
    /// Places `layout` at `offset`, rejecting any tensor whose bytes could leave `bounds`.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::OutsideBounds`] when an element reaches past `bounds`,
    /// [`LayoutError::TensorLayout`] for a dynamic layout or one into Z^n, and
    /// [`LayoutError::Overflow`] when a byte address does not fit in 64 bits.
    pub fn new(
        layout: Layout,
        offset: i64,
        element_size: u64,
        bounds: Bounds,
    ) -> Result<Tensor, LayoutError> {
        let (lower, upper) = byte_range(&layout, offset, element_size)?;
        let inside = lower >= 0 && bounds.start <= lower.unsigned_abs() && upper <= bounds.end;
        if !inside {
            return Err(LayoutError::OutsideBounds {
                lower,
                upper,
                start: bounds.start,
                end: bounds.end,
            });
        }
        let tensor = Tensor { layout, offset, element_size, bounds };
        debug_assert!(tensor.contains());
        Ok(tensor)
    }

    /// Returns the tensor that covers exactly the bytes of its own elements, starting at byte
    /// 0: an array's own storage.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Tensor::new`].
    pub fn over(layout: Layout, element_size: u64) -> Result<Tensor, LayoutError> {
        let (lowest, highest) = element_range(&layout)?;
        let size = i128::from(element_size);
        let end =
            u64::try_from((highest - lowest + 1) * size).map_err(|_| LayoutError::Overflow)?;
        let offset = i64::try_from(-lowest).map_err(|_| LayoutError::Overflow)?;
        Tensor::new(layout, offset, element_size, Bounds { start: 0, end })
    }

    /// Returns whether every element lies inside the bounds, which every constructed tensor
    /// satisfies.
    pub fn contains(&self) -> bool {
        match byte_range(&self.layout, self.offset, self.element_size) {
            Ok((lower, upper)) => {
                lower >= 0 && self.bounds.start <= lower.unsigned_abs() && upper <= self.bounds.end
            }
            Err(_) => false,
        }
    }

    /// Returns the bytes this tensor's own elements occupy.
    pub fn footprint(&self) -> Bounds {
        let (lower, upper) = byte_range(&self.layout, self.offset, self.element_size)
            .expect("a constructed tensor has a representable byte range");
        Bounds { start: lower.unsigned_abs(), end: upper }
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn offset(&self) -> i64 {
        self.offset
    }

    pub fn element_size(&self) -> u64 {
        self.element_size
    }

    pub fn bounds(&self) -> Bounds {
        self.bounds
    }

    /// Returns the subtensor of the free leaves of `coord`, as [`Layout::slice`] defines it.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Layout::slice`] and [`Tensor::new`].
    pub fn slice(&self, coord: &Tuple<Option<Int>>) -> Result<Tensor, LayoutError> {
        let (shift, layout) = self.layout.slice(coord)?;
        let shift = shift.as_int().and_then(Int::as_static).expect("a static layout into Z");
        let offset = self.offset.checked_add(shift).ok_or(LayoutError::Overflow)?;
        Tensor::new(layout, offset, self.element_size, self.bounds)
    }

    /// Returns `self ∘ tiler`, reading the same storage through the composed layout.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Layout::compose`] and [`Tensor::new`].
    pub fn compose(&self, tiler: &Tiler) -> Result<Tensor, LayoutError> {
        let layout = self.layout.compose(tiler)?;
        Tensor::new(layout, self.offset, self.element_size, self.bounds)
    }

    /// Returns the tensor over elements `factor` times as wide, reading the same bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Recast`] when the offset is not aligned to the wider element, and
    /// the errors of [`Layout::recast`] and [`Tensor::new`].
    pub fn recast(&self, factor: i64) -> Result<Tensor, LayoutError> {
        let from = self.element_size;
        let to = u64::try_from(factor)
            .ok()
            .filter(|&factor| factor > 0)
            .and_then(|factor| from.checked_mul(factor))
            .ok_or(LayoutError::Overflow)?;
        if self.offset % factor != 0 {
            let reason = "the offset must be aligned to the new element size";
            return Err(LayoutError::Recast { from, to, reason });
        }
        let layout = self.layout.recast(&Tuple::Leaf(Int::Static(factor)))?;
        Tensor::new(layout, self.offset / factor, to, self.bounds)
    }
}

/// Returns the least and greatest element offset of a static layout into Z, relative to its
/// coordinate 0.
fn element_range(layout: &Layout) -> Result<(i128, i128), LayoutError> {
    let not_static = || LayoutError::TensorLayout { layout: layout.cute().to_string() };
    let (mut lowest, mut highest) = (0i128, 0i128);
    for (extent, step) in layout.shape().extents().into_iter().zip(layout.stride().steps()) {
        let extent = extent.as_static().ok_or_else(not_static)?;
        let step = step.as_int().and_then(Int::as_static).ok_or_else(not_static)?;
        // A mode reaches `(extent - 1) * step` from its first element, below the origin for a
        // negative stride and above it otherwise.
        let reach = i128::from(extent - 1) * i128::from(step);
        if reach < 0 {
            lowest += reach;
        } else {
            highest += reach;
        }
    }
    Ok((lowest, highest))
}

/// Returns the bytes `[lower, upper)` the elements of `layout` at `offset` occupy.
fn byte_range(layout: &Layout, offset: i64, element_size: u64) -> Result<(i64, u64), LayoutError> {
    if element_size == 0 {
        return Err(LayoutError::ZeroElementSize);
    }
    let (lowest, highest) = element_range(layout)?;
    let size = i128::from(element_size);
    let lower = (i128::from(offset) + lowest) * size;
    let upper = (i128::from(offset) + highest + 1) * size;
    let lower = i64::try_from(lower).map_err(|_| LayoutError::Overflow)?;
    // An upper end below zero lies outside every bound, so it saturates to 0 for the check.
    let upper = u64::try_from(upper.max(0)).map_err(|_| LayoutError::Overflow)?;
    Ok((lower, upper))
}
