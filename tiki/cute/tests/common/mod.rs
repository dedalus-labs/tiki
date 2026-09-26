// Copyright © 2026 Dedalus Labs, Inc.

//! Random layouts and the postconditions every layout operation promises.
//!
//! Each check evaluates the layouts involved at every index of their domain and compares the
//! offsets, so it tests the function a layout denotes and never its normal form.

#![allow(dead_code)]

use proptest::prelude::*;
use std::cmp::Ordering;
use tiki_cute::{Int, Layout, Offset, Shape, Stride, Tuple};

/// Largest extent of a random leaf. Four leaves of this extent keep every domain at 256 or
/// fewer coordinates, so the quadratic checks stay fast.
pub const MAX_EXTENT: i64 = 4;
/// Largest stride of a random leaf.
pub const MAX_STRIDE: i64 = 16;
/// Largest number of leaves in a random layout.
pub const MAX_LEAVES: usize = 4;
/// Indices past the end of a complement at which its order is also checked, as PyCuTe does.
pub const PAST_THE_END: i64 = 10;

/// Nests `leaves` by `grouping`: 0 keeps them flat, 1 groups the first two, 2 groups the last
/// two. A grouping that needs more leaves than there are keeps them flat.
pub fn nest<T: Clone>(leaves: &[T], grouping: u8) -> Tuple<T> {
    let leaf = |value: &T| Tuple::Leaf(value.clone());
    let flat = || Tuple::Node(leaves.iter().map(leaf).collect());
    match (grouping, leaves.len()) {
        (_, 1) => leaf(&leaves[0]),
        (1, n) if n >= 3 => {
            let mut modes = vec![Tuple::Node(leaves[..2].iter().map(leaf).collect())];
            modes.extend(leaves[2..].iter().map(leaf));
            Tuple::Node(modes)
        }
        (2, n) if n >= 3 => {
            let mut modes: Vec<Tuple<T>> = leaves[..n - 2].iter().map(leaf).collect();
            modes.push(Tuple::Node(leaves[n - 2..].iter().map(leaf).collect()));
            Tuple::Node(modes)
        }
        _ => flat(),
    }
}

/// Builds a layout with integer strides from leaf extents and strides.
pub fn layout_of(extents: &[i64], strides: &[i64], grouping: u8) -> Layout {
    let extents: Vec<Int> = extents.iter().map(|&e| Int::from(e)).collect();
    let strides: Vec<Offset> = strides.iter().map(|&d| Offset::from(d)).collect();
    let shape = Shape::try_from(nest(&extents, grouping)).expect("positive extents");
    let stride = Stride::try_from(nest(&strides, grouping)).expect("integer strides");
    Layout::new(shape, stride).expect("congruent")
}

prop_compose! {
    /// A random layout of up to four leaves, strides 0 through 16, flat or nested one level.
    pub fn layouts()(leaves in 1..=MAX_LEAVES)(
        extents in prop::collection::vec(1..=MAX_EXTENT, leaves),
        strides in prop::collection::vec(0..=MAX_STRIDE, leaves),
        grouping in 0u8..3,
    ) -> Layout {
        layout_of(&extents, &strides, grouping)
    }
}

prop_compose! {
    /// A random compact layout: the strides of a permutation of the leaves, so it is injective.
    pub fn injective_layouts()(leaves in 1..=MAX_LEAVES)(
        extents in prop::collection::vec(1..=MAX_EXTENT, leaves),
        order in Just((0..leaves).collect::<Vec<usize>>()).prop_shuffle(),
        gap in 1..=3i64,
        grouping in 0u8..3,
    ) -> Layout {
        let mut strides = vec![0; extents.len()];
        let mut running = 1;
        for &leaf in &order {
            strides[leaf] = running;
            running *= extents[leaf] * gap;
        }
        layout_of(&extents, &strides, grouping)
    }
}

/// Returns the static integer size of a layout.
pub fn size(layout: &Layout) -> i64 {
    layout.size().as_static().expect("a static layout has a static size")
}

/// Returns the offset of index `i` as an integer, for layouts into Z.
pub fn at(layout: &Layout, i: i64) -> i64 {
    let offset = layout.at(i);
    offset.as_int().and_then(Int::as_static).expect("a static integer offset")
}

/// Checks `R = A ∘ B`: `B`'s shape is a coordinate shape of `R`, and `R(i) = A(B(i))`.
pub fn check_composition(a: &Layout, b: &Layout, r: &Layout) -> Result<(), String> {
    if !b.shape().is_compatible_with(r.shape()) {
        return Err(format!("{} is not compatible with {}", b.shape(), r.shape()));
    }
    for i in 0..size(r) {
        let through = a.at(at(b, i));
        if r.at(i) != through {
            return Err(format!("R({i}) = {} but A(B({i})) = {through}", r.at(i)));
        }
    }
    Ok(())
}

/// Checks that `R` has `L`'s function on its domain in at most one level of modes.
pub fn check_coalesce(l: &Layout, r: &Layout) -> Result<(), String> {
    if r.depth() > 1 || r.size() != l.size() {
        return Err(format!("{r} is not a flat layout of size {}", l.size()));
    }
    let differs = (0..size(l)).find(|&i| r.at(i) != l.at(i));
    differs.map_or(Ok(()), |i| Err(format!("R({i}) = {} but L({i}) = {}", r.at(i), l.at(i))))
}

/// Checks that `R` is ordered, even past its end, and misses every offset of `L`.
pub fn check_complement(l: &Layout, r: &Layout) -> Result<(), String> {
    let image: Vec<Offset> = (0..size(l)).map(|j| l.at(j)).collect();
    for i in 1..size(r) + PAST_THE_END {
        if r.at(i - 1).compare(&r.at(i)) != Some(Ordering::Less) {
            return Err(format!(
                "R({}) = {} is not below R({i}) = {}",
                i - 1,
                r.at(i - 1),
                r.at(i)
            ));
        }
        if image.contains(&r.at(i)) {
            return Err(format!("R({i}) = {} is in the image of L", r.at(i)));
        }
    }
    Ok(())
}

/// Checks `L(R(i)) = i` over `R`'s domain, for a layout into Z.
pub fn check_right_inverse(l: &Layout, r: &Layout) -> Result<(), String> {
    let differs = (0..size(r)).find(|&i| l.at(at(r, i)) != Offset::from(i));
    differs.map_or(Ok(()), |i| Err(format!("L(R({i})) = {}", l.at(at(r, i)))))
}

/// Checks `L(R(L(i))) = L(i)` over `L`'s domain, for a layout into Z.
pub fn check_left_inverse(l: &Layout, r: &Layout) -> Result<(), String> {
    let differs = (0..size(l)).find(|&i| l.at(at(r, at(l, i))) != l.at(i));
    differs.map_or(Ok(()), |i| Err(format!("L(R(L({i}))) = {}", l.at(at(r, at(l, i))))))
}
