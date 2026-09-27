// Copyright © 2026 Dedalus Labs, Inc.

//! Tilers: the right-hand side of composition, division and product (Cecka, Definition 3.1).

use crate::LayoutError;
use crate::int::Int;
use crate::layout::Layout;
use crate::offset::Offset;
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tuple::Tuple;

/// One layout applied to the whole left layout, or one tiler per left mode.
///
/// A mode tiler of `None` keeps that left mode unchanged. Left modes past the last tiler are
/// dropped by composition and kept by division and product, as in CuTe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tiler {
    /// One layout applied to the whole left layout.
    Layout(Layout),
    /// One tiler per top-level mode of the left layout, `None` to keep a mode as it is.
    Modes(Vec<Option<Tiler>>),
}

impl Tiler {
    /// Returns the single layout that composes the same way (PyCuTe's `tiler_to_layout`).
    ///
    /// The tiler at mode path `p` becomes a layout whose strides point along codomain axis
    /// `E(p)`, so the tiler `(2, 3)` becomes `(2, 3):(1@0, 1@1)`. Composing with that layout
    /// selects the same coordinates of each left mode as the per-mode tilers do.
    pub fn to_layout(&self) -> Result<Layout, LayoutError> {
        self.to_layout_at(&mut Vec::new())
    }

    fn to_layout_at(&self, path: &mut Vec<usize>) -> Result<Layout, LayoutError> {
        match self {
            Tiler::Layout(layout) => scale_into_axis(layout, path),
            Tiler::Modes(modes) => {
                let mut layouts = Vec::with_capacity(modes.len());
                for (i, mode) in modes.iter().enumerate() {
                    let Some(mode) = mode else {
                        return Err(LayoutError::Mismatch {
                            operation: "tiler",
                            detail: format!("mode {i} is unset, so it has no layout"),
                        });
                    };
                    path.push(i);
                    layouts.push(mode.to_layout_at(path)?);
                    path.pop();
                }
                Ok(Layout::from_modes(layouts))
            }
        }
    }
}

/// Moves `layout`'s integer strides onto the codomain axis `E(path)`. The empty path is the
/// integer 1, so a whole-layout tiler keeps its strides, arithmetic tuples and XOR strides
/// included. An arithmetic tuple has integer components, so an XOR stride cannot move onto an
/// axis.
fn scale_into_axis(layout: &Layout, path: &[usize]) -> Result<Layout, LayoutError> {
    if path.is_empty() {
        return Ok(layout.clone());
    }
    let axis = Offset::basis(path);
    let mut stride = Vec::new();
    for d in layout.stride().steps() {
        if d.as_xor().is_some() {
            return Err(LayoutError::XorStride { operation: "tiler", stride: d.to_string() });
        }
        let Some(d) = d.as_int() else {
            return Err(LayoutError::Mismatch {
                operation: "tiler",
                detail: format!("stride {d} of {layout} is already an arithmetic tuple"),
            });
        };
        stride.push(axis.scale(d));
    }
    let stride = Tuple::from_leaves(&mut stride.into_iter(), layout.stride().as_tuple());
    Ok(Layout::from_parts(layout.shape().clone(), Stride::from_derived(stride)))
}

impl Layout {
    /// Applies `op` to each mode with its tiler. Modes without a tiler, or past the last one,
    /// stay as they are. Division and product share this rule. Composition drops the modes
    /// past the last tiler instead.
    pub(crate) fn each_mode(
        &self,
        operation: &'static str,
        tilers: &[Option<Tiler>],
        op: impl Fn(&Layout, &Tiler) -> Result<Layout, LayoutError>,
    ) -> Result<Layout, LayoutError> {
        if self.rank() < tilers.len() {
            return Err(LayoutError::TilerRank {
                operation,
                tiler: tilers.len(),
                layout: self.rank(),
            });
        }
        let modes = self.modes().enumerate().map(|(i, mode)| match tilers.get(i) {
            Some(Some(tiler)) => op(&mode, tiler),
            _ => Ok(mode),
        });
        Ok(Layout::from_modes(modes.collect::<Result<Vec<_>, _>>()?))
    }
}

/// Nests per-mode layouts into a tiler of the same shape.
impl From<Tuple<Layout>> for Tiler {
    fn from(layouts: Tuple<Layout>) -> Self {
        match layouts {
            Tuple::Leaf(layout) => Tiler::Layout(layout),
            Tuple::Node(modes) => {
                Tiler::Modes(modes.into_iter().map(|mode| Some(Tiler::from(mode))).collect())
            }
        }
    }
}

impl From<Layout> for Tiler {
    fn from(layout: Layout) -> Self {
        Tiler::Layout(layout)
    }
}

/// An extent `n` tiles with the contiguous layout `n:1`.
impl TryFrom<Int> for Tiler {
    type Error = LayoutError;

    fn try_from(extent: Int) -> Result<Self, LayoutError> {
        Ok(Tiler::Layout(Layout::from(Shape::try_from(Tuple::Leaf(extent))?)))
    }
}

impl TryFrom<i64> for Tiler {
    type Error = LayoutError;

    fn try_from(extent: i64) -> Result<Self, LayoutError> {
        Tiler::try_from(Int::Static(extent))
    }
}

impl<const N: usize> From<[Option<Tiler>; N]> for Tiler {
    fn from(modes: [Option<Tiler>; N]) -> Self {
        Tiler::Modes(modes.into())
    }
}
