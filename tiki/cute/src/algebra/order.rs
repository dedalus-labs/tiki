// Copyright © 2026 Dedalus Labs, Inc.

//! Compact layouts whose modes nest in a chosen order.

use crate::LayoutError;
use crate::int::Int;
use crate::layout::Layout;
use crate::offset::{Offset, static_first};
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tuple::Tuple;

impl Layout {
    /// Returns the compact layout of `shape` whose leaves nest in `order`: the leaf with the
    /// smallest order steps by 1, the next by the extent of the first, and so on (CuTe's
    /// `make_ordered_layout`).
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Mismatch`] when `order` is not congruent to `shape`.
    pub fn from_order(shape: Shape, order: &Tuple<Int>) -> Result<Layout, LayoutError> {
        if !shape.as_tuple().congruent(order) {
            return Err(LayoutError::Mismatch {
                operation: "ordered layout",
                detail: format!("order {order} is not congruent to shape {shape}"),
            });
        }
        let keys: Vec<Offset> = order.leaves().into_iter().cloned().map(Offset::from).collect();
        let stride = compact_in_order(&shape, &keys, |_| true);
        Ok(Layout::from_parts(shape, stride))
    }

    /// Returns the compact layout of this shape whose leaves nest in the order of this layout's
    /// strides. Broadcast leaves, with stride 0, stay broadcast (CuTe's `make_layout_like`).
    #[must_use]
    pub fn compact_like(&self) -> Layout {
        let keys: Vec<Offset> = self.stride().steps().into_iter().cloned().collect();
        let stride = compact_in_order(self.shape(), &keys, |key| !key.is_zero());
        Layout::from_parts(self.shape().clone(), stride)
    }
}

/// Assigns compact strides to the leaves of `shape` in the order of `keys`, one key per leaf.
/// Leaves whose key fails `strided` keep stride 0 and take no room.
fn compact_in_order(shape: &Shape, keys: &[Offset], strided: impl Fn(&Offset) -> bool) -> Stride {
    let extents = shape.extents();
    let mut order: Vec<usize> = (0..extents.len()).collect();
    order.sort_by(|&a, &b| static_first(&keys[a], &keys[b]));

    let mut steps = vec![Offset::zero(); extents.len()];
    let mut running = Int::Static(1);
    for leaf in order.into_iter().filter(|&leaf| strided(&keys[leaf])) {
        steps[leaf] = Offset::from(running.clone());
        running = &running * extents[leaf];
    }
    Stride::from_derived(Tuple::from_leaves(&mut steps.into_iter(), shape.as_tuple()))
}
