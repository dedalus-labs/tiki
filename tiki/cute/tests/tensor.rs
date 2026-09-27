// Copyright © 2026 Dedalus Labs, Inc.

//! Checks every tensor rule against a brute-force walk of each element's bytes.

mod common;

use common::*;
use proptest::prelude::*;
use std::collections::BTreeSet;
use tiki_cute::{Bounds, Int, LayoutError, Tensor, Tiler, Tuple};

/// Largest stride magnitude of a random tensor layout. Negative strides are included, since
/// they place elements below the origin.
const MAX_STEP: i64 = 6;
const MAX_OFFSET: i64 = 40;
const MAX_BOUNDS_START: u64 = 64;
const MAX_BOUNDS_LENGTH: u64 = 256;
const ELEMENT_SIZES: [u64; 4] = [1, 2, 4, 8];
const CASES_PER_PROPERTY: u32 = 4096;
/// A 4-byte element, for the derived-tensor properties.
const FLOAT: u64 = 4;

/// Returns the byte address of every element of `tensor`.
fn byte_addresses(tensor: &Tensor) -> BTreeSet<i64> {
    let layout = tensor.layout();
    let size = i64::try_from(tensor.element_size()).unwrap();
    (0..common::size(layout)).map(|i| (tensor.offset() + at(layout, i)) * size).collect()
}

prop_compose! {
    fn placements()(leaves in 1..=MAX_LEAVES)(
        extents in prop::collection::vec(1..=MAX_EXTENT, leaves),
        steps in prop::collection::vec(-MAX_STEP..=MAX_STEP, leaves),
        grouping in 0u8..3,
        offset in -10i64..MAX_OFFSET,
        element_size in prop::sample::select(ELEMENT_SIZES.to_vec()),
        start in 0..MAX_BOUNDS_START,
        length in 0..MAX_BOUNDS_LENGTH,
    ) -> (Vec<i64>, Vec<i64>, u8, i64, u64, Bounds) {
        (extents, steps, grouping, offset, element_size, Bounds { start, end: start + length })
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES_PER_PROPERTY))]

    // Invariant: Tensor::new accepts a tensor exactly when every element's bytes lie in bounds.
    // Witness: the brute-force walk agrees with the constructor on every random placement.
    #[test]
    fn construction_accepts_exactly_the_tensors_inside_their_bounds(
        (extents, steps, grouping, offset, element_size, bounds) in placements()
    ) {
        let layout = layout_of(&extents, &steps, grouping);
        let size = i64::try_from(element_size).unwrap();
        let bytes: Vec<i64> = (0..common::size(&layout)).map(|i| (offset + at(&layout, i)) * size).collect();
        let inside = bytes.iter().all(|&byte| {
            byte >= 0 && bounds.start <= byte.unsigned_abs() && byte.unsigned_abs() + element_size <= bounds.end
        });
        let built = Tensor::new(layout, offset, element_size, bounds);
        prop_assert_eq!(built.is_ok(), inside, "{:?}", built);
        if let Ok(tensor) = built {
            prop_assert!(tensor.contains());
        }
    }

    // Invariant: slicing and composing never leave the source's storage, and a tile inside the
    // source's domain reads only the source's own elements. A tile past the domain may read
    // other bytes of the storage, which a kernel that masks those lanes is free to do.
    // Witness: every derived tensor is inside the bounds, and an in-domain tile's bytes are a
    // subset of the source's bytes.
    #[test]
    fn derived_tensors_stay_within_their_source(source in injective_layouts(), fixed in 0..MAX_EXTENT, tile in layouts()) {
        let tensor = Tensor::over(source.clone(), FLOAT).unwrap();
        let reach = byte_addresses(&tensor);

        let first = source.shape().as_tuple().modes()[0].leaves()[0].as_static().unwrap();
        let mut coord: Vec<Tuple<Option<Int>>> = (0..source.rank()).map(|_| Tuple::Leaf(None)).collect();
        if source.depth() == 1 && source.shape().as_tuple().modes()[0].depth() == 0 {
            coord[0] = Tuple::Leaf(Some(Int::from(fixed % first)));
            let sliced = tensor.slice(&Tuple::Node(coord)).unwrap();
            prop_assert!(byte_addresses(&sliced).is_subset(&reach));
        }

        let in_domain = (0..common::size(&tile)).all(|i| at(&tile, i) < common::size(&source));
        if let Ok(composed) = tensor.compose(&Tiler::from(tile)) {
            prop_assert!(composed.contains());
            prop_assert!(!in_domain || byte_addresses(&composed).is_subset(&reach));
        }
    }
}

// Invariant: recasting covers the same bytes and rejects offsets misaligned for the wider type.
// Witness: bytes 0..8 of byte storage become two 4-byte elements. Bytes 1..9 are refused.
#[test]
fn recast_keeps_the_same_bytes_and_checks_alignment() {
    const STORAGE: &str = "9:1";
    const WIDEN: i64 = 4;
    let bytes = Tensor::over(STORAGE.parse().unwrap(), 1).unwrap();
    for (start, expected) in [(0, Some("2:1")), (1, None)] {
        let window = bytes.compose(&"8:1".parse::<Tiler>().unwrap()).unwrap();
        let window = Tensor::new(window.layout().clone(), start, 1, bytes.bounds()).unwrap();
        let recast = window.recast(WIDEN);
        let got = recast.as_ref().ok().map(|t| t.layout().cute().to_string());
        assert_eq!(got.as_deref(), expected, "bytes {start}..{}: {recast:?}", start + 8);
    }
}

// Invariant: a layout the byte check cannot evaluate is refused before any address exists.
// Witness: a dynamic extent and an arithmetic-tuple stride are both refused.
#[test]
fn tensors_need_static_integer_layouts() {
    for layout in ["N:1", "4:1@0"] {
        let refused = Tensor::over(layout.parse().unwrap(), FLOAT);
        assert!(matches!(refused, Err(LayoutError::TensorLayout { .. })), "{layout}: {refused:?}");
    }
}
