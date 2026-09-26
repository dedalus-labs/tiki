// Copyright © 2026 Dedalus Labs, Inc.

//! Every operation's postconditions on random static layouts.
//!
//! The reference cases prove agreement with PyCuTe on the layouts its authors chose. These
//! properties prove each operation's contract on layouts nobody chose, by evaluating every
//! index. An operation may refuse a layout that breaks its preconditions. Any layout it returns
//! must satisfy the contract.

mod common;

use common::*;
use proptest::prelude::*;
use tiki_cute::{Layout, Tiler};

/// Random cases per property. Each case evaluates at most a few thousand offsets.
const CASES: u32 = 2048;
/// Draws a property may discard. Composition keeps only right layouts whose image lies in the
/// left layout's domain, which most random pairs miss.
const REJECTS: u32 = 1 << 20;

proptest! {
    #![proptest_config(ProptestConfig { cases: CASES, max_global_rejects: REJECTS, ..ProptestConfig::default() })]

    // Invariant: coalescing keeps the function and flattens the layout, and does it once.
    // Witness: every index of the original and the coalesced layout maps to the same offset.
    #[test]
    fn coalesce_keeps_the_function(l in layouts()) {
        let r = l.coalesce();
        prop_assert_eq!(check_coalesce(&l, &r), Ok(()), "{} -> {}", l.cute(), r.cute());
        prop_assert_eq!(r.coalesce(), r);
    }

    // Invariant: a composition that succeeds evaluates A at B's offsets, wherever B's offsets
    // are coordinates of A. Past A's domain, CuTe and PyCuTe extend A differently per operand,
    // so the contract stops at A's size.
    // Witness: R(i) = A(B(i)) at every index of R.
    #[test]
    fn composition_evaluates_a_at_b(a in layouts(), b in layouts()) {
        let reach = (0..size(&b)).map(|i| at(&b, i)).max().unwrap_or(0);
        prop_assume!(reach < size(&a));
        if let Ok(r) = a.compose(&Tiler::from(b.clone())) {
            prop_assert_eq!(check_composition(&a, &b, &r), Ok(()), "{} o {} -> {}", a.cute(), b.cute(), r.cute());
        }
    }

    // Invariant: composing with the contiguous layout of A's whole domain is coalescing.
    // Witness: A o size(A):1 equals coalesce(A), structurally.
    #[test]
    fn composition_with_the_whole_domain_coalesces(a in layouts()) {
        let whole = layout_of(&[size(&a)], &[1], 0);
        let r = a.compose(&Tiler::from(whole)).expect("the whole domain always composes");
        prop_assert_eq!(r, a.coalesce(), "{}", a.cute());
    }

    // Invariant: the complement of an injective layout is ordered and disjoint from it.
    // Witness: consecutive complement offsets increase, and none is an offset of L.
    #[test]
    fn complement_fills_the_gaps(l in injective_layouts()) {
        let r = l.complement().expect("an injective layout has a complement");
        prop_assert_eq!(check_complement(&l, &r), Ok(()), "{} -> {}", l.cute(), r.cute());
    }

    // Invariant: the right inverse inverts the contiguous prefix of the image.
    // Witness: L(R(i)) = i for every index of R.
    #[test]
    fn right_inverse_inverts_the_image_prefix(l in layouts()) {
        let r = l.right_inverse();
        prop_assert_eq!(check_right_inverse(&l, &r), Ok(()), "{} -> {}", l.cute(), r.cute());
    }

    // Invariant: a left inverse that succeeds recovers every coordinate from its offset.
    // Witness: L(R(L(i))) = L(i) for every index of L.
    #[test]
    fn left_inverse_recovers_coordinates(l in layouts()) {
        if let Ok(r) = l.left_inverse() {
            prop_assert_eq!(check_left_inverse(&l, &r), Ok(()), "{} -> {}", l.cute(), r.cute());
        }
    }

    // Invariant: an injective layout always has a left inverse.
    // Witness: every compact layout with gaps succeeds.
    #[test]
    fn injective_layouts_have_left_inverses(l in injective_layouts()) {
        let r = l.left_inverse().expect("an injective layout has a left inverse");
        prop_assert_eq!(check_left_inverse(&l, &r), Ok(()), "{} -> {}", l.cute(), r.cute());
    }

    // Invariant: a product holds one copy of A per coordinate of B, and mode 0 is A itself.
    // Witness: rank 2, size(A) * size(B), R[0] = A, and each copy lies in A's complement.
    #[test]
    fn logical_product_repeats_a(a in injective_layouts(), b in injective_layouts()) {
        let Ok(r) = a.logical_product(&Tiler::from(b.clone())) else { return Ok(()) };
        prop_assert_eq!(r.rank(), 2);
        prop_assert_eq!(size(&r), size(&a) * size(&b));
        prop_assert_eq!(r.get(&[0]), Some(a.clone()));
        prop_assert!(b.shape().is_compatible_with(r.get(&[1]).unwrap().shape()));
    }

    // Invariant: zipped division splits A into tiles, and a division that covers A's size loses
    // no offset. A tile that does not divide A leaves a partial last tile outside the grid,
    // which a kernel covers with predication.
    // Witness: R(i, 0) = A(B(i)), and when size(R) = size(A) every offset of A is in R.
    #[test]
    fn zipped_divide_tiles_a(a in injective_layouts(), b in injective_layouts()) {
        let reach = (0..size(&b)).map(|i| at(&b, i)).max().unwrap_or(0);
        prop_assume!(reach < size(&a));
        let Ok(r) = a.zipped_divide(&Tiler::from(b.clone())) else { return Ok(()) };
        prop_assert_eq!(r.rank(), 2);
        let tile = r.get(&[0]).unwrap();
        prop_assert_eq!(check_composition(&a, &b, &tile), Ok(()));
        prop_assume!(size(&r) == size(&a));
        let offsets: Vec<_> = (0..size(&r)).map(|i| r.at(i)).collect();
        for i in 0..size(&a) {
            prop_assert!(offsets.contains(&a.at(i)), "A({}) missing from {}", i, r.cute());
        }
    }
}

// Every constructor form prints back as it was written.
#[test]
fn printing_round_trips_through_parsing() {
    let cases = [
        "4:1",
        "(4, 3):(3, 1)",
        "((2, 2), (2, 2)):((4, 8), (1, 2))",
        "(4,):(2,)",
        "(3, 4):(1@0, 2@1)",
    ];
    for text in cases {
        let layout: Layout = text.parse().unwrap();
        assert_eq!(layout.cute().to_string(), text, "{text}");
    }
}
