// Copyright © 2026 Dedalus Labs, Inc.

//! XOR strides: each operation's postconditions on random XOR layouts, the equivalence of a
//! [`Swizzle`] and its XOR layout, and the cases where tiki-cute deliberately differs from
//! PyCuTe's `F2` paths, each with the index that proves PyCuTe's result wrong.

mod common;

use common::*;
use proptest::prelude::*;
use tiki_cute::{
    Condition, Int, Layout, LayoutError, Offset, Param, Shape, Stride, Swizzle, SwizzleParams,
    Tiler, Tuple, Verdict, Xor,
};

/// Random cases per property. Each case evaluates at most a few thousand offsets.
const CASES: u32 = 2048;
/// Draws a property may discard, as in `postconditions.rs`.
const REJECTS: u32 = 1 << 20;

fn layout(text: &str) -> Layout {
    text.parse().unwrap()
}

fn bits(value: i64) -> Xor {
    Xor::try_from(value).unwrap()
}

/// Returns whether `result` refused with the carry-free condition violated.
fn carries(result: &Result<Layout, LayoutError>) -> bool {
    matches!(
        result,
        Err(LayoutError::Condition {
            condition: Condition::CarryFree,
            verdict: Verdict::Violated,
            ..
        })
    )
}

proptest! {
    #![proptest_config(ProptestConfig { cases: CASES, max_global_rejects: REJECTS, ..ProptestConfig::default() })]

    // Invariant: coalescing an XOR layout keeps its function inside the domain and, for
    // coalesce_z, past it, and coalescing twice changes nothing.
    // Witness: every index of the original and the coalesced layout maps to the same offset.
    #[test]
    fn xor_coalesce_keeps_the_function(l in xor_layouts()) {
        let r = l.coalesce();
        prop_assert_eq!(check_coalesce(&l, &r), Ok(()), "{} -> {}", l.cute(), r.cute());
        prop_assert_eq!(r.coalesce(), r);
        let z = l.coalesce_z();
        let differs = (0..2 * size(&l)).find(|&i| z.at(i) != l.at(i));
        prop_assert_eq!(differs, None, "{} -> {}", l.cute(), z.cute());
    }

    // Invariant: a composition with an XOR left layout that succeeds evaluates A at B's
    // offsets, wherever they are coordinates of A.
    // Witness: R(i) = A(B(i)) at every index of R.
    #[test]
    fn xor_composition_evaluates_a_at_b(a in xor_layouts(), b in layouts()) {
        let reach = (0..size(&b)).map(|i| at(&b, i)).max().unwrap_or(0);
        prop_assume!(reach < size(&a));
        if let Ok(r) = a.compose(&Tiler::from(b.clone())) {
            prop_assert_eq!(check_composition(&a, &b, &r), Ok(()), "{} o {} -> {}", a.cute(), b.cute(), r.cute());
        }
    }

    // Invariant: composing an XOR layout with its whole domain is coalescing, and never refuses.
    // Witness: A o size(A):1 equals coalesce(A), structurally.
    #[test]
    fn xor_composition_with_the_whole_domain_coalesces(a in xor_layouts()) {
        let whole = layout_of(&[size(&a)], &[1], 0);
        let r = a.compose(&Tiler::from(whole)).expect("the whole domain always composes");
        prop_assert_eq!(r, a.coalesce(), "{}", a.cute());
    }

    // Invariant: a composition with an XOR right layout that succeeds evaluates A at B's XOR
    // offsets read as XOR indices.
    // Witness: R(i) = A(B(i)) at every index whose B(i) lies below size(A).
    #[test]
    fn composition_with_xor_b_evaluates_a_at_xor_indices(a in layouts(), b in xor_layouts()) {
        if let Ok(r) = a.compose(&Tiler::from(b.clone())) {
            for i in 0..size(&b) {
                let through = b.at(i);
                let reach = through.as_xor().map_or(0, Xor::value);
                if reach < size(&a) {
                    prop_assert_eq!(r.at(i), a.call_offset(&through).unwrap(), "{} o {} at {}", a.cute(), b.cute(), i);
                }
            }
        }
    }

    // Invariant: the right inverse of an XOR layout inverts it on XOR indices.
    // Witness: L(R(k)) = ^k for every index k of R.
    #[test]
    fn xor_right_inverse_inverts_the_image_prefix(l in xor_layouts()) {
        let r = l.right_inverse();
        for k in 0..size(&r) {
            prop_assert_eq!(l.call_offset(&r.at(k)), Ok(xor(k)), "{} -> {}", l.cute(), r.cute());
        }
    }

    // Invariant: on power-of-two extents an XOR index splits like the integer index, and the
    // right inverse keeps PyCuTe's whole chain, which then satisfies the generalized contract.
    // Witness: L(^i) = L(i) for every index, and R(L(R(k))) = R(k).
    #[test]
    fn pow2_xor_layouts_split_like_integers(l in pow2_xor_layouts()) {
        for i in 0..size(&l) {
            prop_assert_eq!(l.call_offset(&xor(i)), Ok(l.at(i)), "{} at {}", l.cute(), i);
        }
        let r = l.right_inverse();
        for k in 0..size(&r) {
            let back = r.call_offset(&l.call_offset(&r.at(k)).unwrap());
            prop_assert_eq!(back, Ok(r.at(k)), "{} -> {}", l.cute(), r.cute());
        }
    }

    // Invariant: a left inverse of an XOR layout that succeeds recovers every coordinate.
    // Witness: L(R(L(i))) = L(i) for every index of L.
    #[test]
    fn xor_left_inverse_recovers_coordinates(l in xor_layouts()) {
        if let Ok(r) = l.left_inverse() {
            for i in 0..size(&l) {
                let back = l.call_offset(&r.call_offset(&l.at(i)).unwrap()).unwrap();
                prop_assert_eq!(back, l.at(i), "{} -> {}", l.cute(), r.cute());
            }
        }
    }

    // Invariant: a single stride-^1 mode among zero modes has a left inverse on power-of-two
    // extents, where the XOR split is the integer split.
    // Witness: every placement of the ^1 mode among up to three broadcast modes.
    #[test]
    fn a_unit_xor_mode_has_a_left_inverse(
        extents in prop::collection::vec(prop::sample::select(POWERS_OF_TWO.to_vec()), 1..=MAX_LEAVES),
        unit in 0..MAX_LEAVES,
    ) {
        let unit = unit % extents.len();
        let strides: Vec<i64> = (0..extents.len()).map(|i| i64::from(i == unit)).collect();
        let l = xor_layout_of(&extents, &strides, 0);
        let r = l.left_inverse().expect("a unit XOR mode inverts");
        for i in 0..size(&l) {
            prop_assert_eq!(l.call_offset(&r.call_offset(&l.at(i)).unwrap()), Ok(l.at(i)));
        }
    }

    // Invariant: an XOR layout sum that succeeds is the pointwise XOR of the summands.
    // Witness: R(i) = A(i) + B(i) at every index, for summands over one set of extents.
    #[test]
    fn xor_layout_addition_is_pointwise(
        extents in prop::collection::vec(1..=MAX_EXTENT, 1..=MAX_LEAVES),
        left in prop::collection::vec(0..=MAX_XOR_BITS, MAX_LEAVES),
        right in prop::collection::vec(0..=MAX_XOR_BITS, MAX_LEAVES),
        grouping in 0u8..3,
    ) {
        let a = xor_layout_of(&extents, &left[..extents.len()], grouping);
        let b = xor_layout_of(&extents, &right[..extents.len()], 0);
        if let Ok(r) = a.add(&b) {
            for i in 0..size(&a) {
                prop_assert_eq!(r.at(i), a.at(i).add(&b.at(i)), "{} + {} -> {}", a.cute(), b.cute(), r.cute());
            }
        }
    }

    // Invariant: a zipped division of an XOR layout by an integer tile that succeeds holds the
    // tile in mode 0.
    // Witness: R(i, 0) = A(B(i)) at every index of the tile.
    #[test]
    fn xor_zipped_divide_tiles_a(a in pow2_xor_layouts(), b in injective_layouts()) {
        let reach = (0..size(&b)).map(|i| at(&b, i)).max().unwrap_or(0);
        prop_assume!(reach < size(&a));
        if let Ok(r) = a.zipped_divide(&Tiler::from(b.clone())) {
            let tile = r.get(&[0]).unwrap();
            prop_assert_eq!(check_composition(&a, &b, &tile), Ok(()), "{} / {}", a.cute(), b.cute());
        }
    }

    // Invariant: the coshape of an XOR layout bounds its image, as PyCuTe's does.
    // Witness: every offset's bits lie below the coshape.
    #[test]
    fn xor_coshape_bounds_the_image(l in xor_layouts()) {
        let bound = l.coshape();
        let Tuple::Leaf(bound) = bound else { panic!("an XOR codomain is a leaf") };
        let bound = bound.as_static().unwrap();
        for i in 0..size(&l) {
            let value = l.at(i).as_xor().map_or(0, Xor::value);
            prop_assert!(value < bound, "{} reaches {} past {}", l.cute(), value, bound);
        }
    }

    // Invariant: an XOR layout printed in CuTe notation parses back to the same layout.
    // Witness: every random XOR layout and its coalesced form and right inverse.
    #[test]
    fn printed_xor_layouts_parse_back(l in xor_layouts()) {
        for derived in [l.clone(), l.coalesce(), l.right_inverse()] {
            let printed = derived.cute().to_string();
            prop_assert_eq!(printed.parse::<Layout>(), Ok(derived), "{}", printed);
        }
    }
}

// Invariant: a Swizzle and the XOR layout with one mode per index bit are the same function.
// Witness: every valid swizzle with fields up to 3 bits wide, at every index of every
// power-of-two domain that holds both fields, coalesced or not.
#[test]
fn swizzle_and_its_xor_layout_agree_on_every_index() {
    for bits in 0..=3 {
        for base in 0..=3 {
            for shift in -6..=6_i64 {
                let params = SwizzleParams { bits, base, shift };
                let Ok(swizzle) = Swizzle::try_from(params) else { continue };
                let width = base + bits + shift.abs();
                for domain in width..=width + 1 {
                    let l = swizzle_layout(&swizzle, domain);
                    let c = l.coalesce();
                    for index in 0..1 << domain {
                        let swizzled = xor(swizzle.apply(index).unwrap());
                        assert_eq!(l.at(index), swizzled, "{swizzle} over 2^{domain} at {index}");
                        assert_eq!(c.at(index), swizzled, "{swizzle} as {} at {index}", c.cute());
                    }
                }
            }
        }
    }
    let canonical = Swizzle::try_from(SwizzleParams { bits: 3, base: 0, shift: 3 }).unwrap();
    assert_eq!(swizzle_layout(&canonical, 6).coalesce(), layout("(8, 8):(^1, ^9)"));
}

/// Returns the XOR layout of `swizzle` over indices below `2^domain`: one extent-2 mode per bit,
/// whose stride is the bit itself plus the destination bit it is copied onto, if any.
fn swizzle_layout(swizzle: &Swizzle, domain: i64) -> Layout {
    let (width, base, shift) =
        (i64::from(swizzle.bits()), i64::from(swizzle.base()), i64::from(swizzle.shift()));
    let source = if shift >= 0 { base + shift } else { base };
    let strides: Vec<i64> = (0..domain)
        .map(|bit| {
            let copied = (source..source + width).contains(&bit);
            (1 << bit) ^ if copied { 1 << (bit - shift) } else { 0 }
        })
        .collect();
    let extents = vec![2; strides.len()];
    if extents.is_empty() {
        return layout("1:0");
    }
    xor_layout_of(&extents, &strides, 0)
}

// Invariant: an XOR index splits over a shape exactly when every extent multiplies the prefix
// product before it without a carry, and then recombines to itself, as in PyCuTe.
// Witness: every rank-3 shape with extents 1 through 6, at every index in and past its domain.
#[test]
fn xor_index_splits_exactly_on_carry_free_shapes() {
    for s0 in 1..=6 {
        for s1 in 1..=6 {
            for s2 in 1..=6_i64 {
                let shape: Shape = format!("({s0}, {s1}, {s2})").parse().unwrap();
                let carry_free = (bits(s0) * bits(s1)).value() == s0 * s1;
                for i in 0..2 * s0 * s1 * s2 {
                    let split = shape.idx2crd_xor(bits(i));
                    if !carry_free {
                        assert!(
                            matches!(
                                split,
                                Err(LayoutError::Condition { condition: Condition::CarryFree, .. })
                            ),
                            "{shape} splits ^{i}"
                        );
                        continue;
                    }
                    assert_eq!(shape.crd2idx_xor(&split.unwrap()), Ok(bits(i)), "{shape} at {i}");
                }
            }
        }
    }
}

// Invariant: on power-of-two extents the carry-less split is the integer split.
// Witness: PyCuTe's examples, and every index of its power-of-two shapes.
#[test]
fn xor_index_splits_bits_on_power_of_two_extents() {
    let shape: Shape = "(4, 8)".parse().unwrap();
    let split = Tuple::node([Tuple::Leaf(bits(0b10)), Tuple::Leaf(bits(0b101))]);
    assert_eq!(shape.idx2crd_xor(bits(0b1_0110)), Ok(split));
    let past = Tuple::node([Tuple::Leaf(bits(0b10)), Tuple::Leaf(bits(0b1101))]);
    assert_eq!(shape.idx2crd_xor(bits(0b11_0110)), Ok(past));
    let leaf: Shape = "32".parse().unwrap();
    assert_eq!(leaf.idx2crd_xor(bits(22)), Ok(Tuple::Leaf(bits(22))));
    for text in ["(4, 8)", "(2, 4, 4)", "(2, (4, 4))", "(8, 8)", "((2, 2), (4, 2))"] {
        let shape: Shape = text.parse().unwrap();
        let size = shape.size().as_static().unwrap();
        for i in 0..size {
            let integer = shape.idx2crd(&Tuple::Leaf(Int::from(i))).unwrap();
            let carry_less = shape.idx2crd_xor(bits(i)).unwrap();
            assert_eq!(
                carry_less.to_string().replace('^', ""),
                integer.to_string(),
                "{text} at {i}"
            );
        }
    }
}

// Invariant: XOR strides evaluate by XOR and carry-less products, as PyCuTe's F2 does.
// Witness: PyCuTe's examples, including 3 * ^3 = ^5 where 3 * 3 = 9.
#[test]
fn xor_layouts_evaluate_carry_lessly() {
    let l = layout("(4, 8):(^1, ^8)");
    for c0 in 0..4 {
        for c1 in 0..8 {
            let coord = Tuple::node([Tuple::Leaf(Int::from(c0)), Tuple::Leaf(Int::from(c1))]);
            assert_eq!(l.call(&coord), Ok(xor(c0 ^ (8 * c1))));
        }
    }
    assert_eq!(layout("4:^3").at(3), xor(0b101));
    assert_eq!(layout("(4, 4):(^1, ^4)").at(5).add(&layout("(4, 4):(^1, ^4)").at(5)), xor(0));
    assert_eq!(layout("(8, 8):(^1, ^9)").coshape(), Tuple::Leaf(Int::from(64)));
    assert_eq!(layout("4:^3").coshape(), Tuple::Leaf(Int::from(8)));
    let compact = layout("(4, 8):(^1, ^4)");
    for i in 0..32 {
        assert_eq!(compact.call_offset(&xor(i)), Ok(xor(i)), "index {i}");
    }
}

// Invariant: an XOR stride prints as ^v, parses back, and never reads as a parameter name or
// sits where an integer is required.
// Witness: round trips, the zero ^0, and each malformed use.
#[test]
fn xor_notation_round_trips_and_stays_apart() {
    for text in ["(8, 8):(^1, ^9)", "16:^3", "(4, (4, 3)):(^1, (^5, ^16))", "(4, 8):(^1, 0)"] {
        assert_eq!(layout(text).cute().to_string(), text);
    }
    assert_eq!(layout("(4, 8):(^1, ^4)").to_string(), "Layout(shape=(4, 8), stride=(^1, ^4))");
    assert_eq!(layout("16:^0"), layout("16:0"));
    assert!(
        layout("(4, 8):(^1, ^4)").describe().to_string().ends_with("index = ^1 * c[0] + ^4 * c[1]")
    );
    let named: Layout = "4:F2".parse().unwrap();
    assert!(!named.stride().is_xor(), "F2 is a parameter name");
    for text in ["4:^-1", "4:-^1", "4:^1+^2", "4:2*^1", "4:^N", "(^2, 2):(1, 2)"] {
        assert!(text.parse::<Layout>().is_err(), "{text}");
    }
    assert!("^4".parse::<Tiler>().is_err());
    assert!(matches!("(2, 2):(1, ^2)".parse::<Layout>(), Err(LayoutError::MixedCodomain { .. })));
    assert!(matches!("(2, 2):(1@0, ^2)".parse::<Layout>(), Err(LayoutError::MixedCodomain { .. })));
}

// Invariant: an extent or coordinate that an XOR stride multiplies is static and nonnegative.
// Witness: a dynamic extent at construction, and a negative or dynamic index at evaluation.
#[test]
fn xor_strides_refuse_integers_without_bits() {
    let n = Int::from(Param::positive("N"));
    let shape = Shape::try_from(Tuple::Leaf(n.clone())).unwrap();
    let stride = Stride::try_from(Tuple::Leaf(xor(1))).unwrap();
    assert!(matches!(Layout::new(shape, stride), Err(LayoutError::XorOperand { .. })));
    assert!(matches!("N:^1".parse::<Layout>(), Err(LayoutError::XorOperand { .. })));
    let l = layout("(4, 4):(^1, ^4)");
    assert!(matches!(l.call(&Tuple::Leaf(Int::from(-1))), Err(LayoutError::XorOperand { .. })));
    assert!(matches!(l.call(&Tuple::Leaf(n)), Err(LayoutError::XorOperand { .. })));
    let dynamic = layout("(N, 2):(0, 0)");
    let modes = [dynamic, layout("2:^1")];
    assert!(matches!(Layout::try_from_modes(modes), Err(LayoutError::XorOperand { .. })));
    assert!(matches!(Xor::try_from(-3), Err(LayoutError::XorOperand { .. })));
}

// Invariant: operations PyCuTe has no F2 rule for refuse with a typed error.
// Witness: complement, recast of a stride other than ^1, a walk of an XOR stride across modes,
// and an XOR tiler moved onto a codomain axis.
#[test]
fn xor_strides_without_a_rule_are_refused() {
    let refused =
        |result: Result<Layout, LayoutError>| matches!(result, Err(LayoutError::XorStride { .. }));
    assert!(refused(layout("8:^1").complement()));
    assert_eq!(layout("(1, 4):(^3, 0)").complement(), layout("(1, 4):(0, 0)").complement());
    assert!(refused(layout("8:^2").recast(&"2".parse().unwrap())));
    assert_eq!(layout("8:^1").recast(&"2".parse().unwrap()), Ok(layout("4:^1")));
    assert!(refused(layout("(4, 8):(1, 8)").compose(&Tiler::from(layout("4:^1")))));
    assert_eq!(layout("(4, 8):(1, 8)").compose(&Tiler::from(layout("8:^4"))), Ok(layout("8:^8")));
    assert_eq!(layout("8:1").compose(&Tiler::from(layout("4:^3"))), Ok(layout("4:^3")));
    let tiler: Tiler =
        [Some(Tiler::from(layout("2:^1"))), Some(Tiler::try_from(2).unwrap())].into();
    assert!(matches!(tiler.to_layout(), Err(LayoutError::XorStride { .. })));
}

// Invariant: a composition whose integer walk carries a bit through an XOR stride refuses.
// PyCuTe multiplies the XOR stride by the walk's stride and returns 16:^1 o 4:3 = 4:^3.
// Witness: A(B(3)) = A(9) = ^9, but PyCuTe's 4:^3 maps 3 to 3 * ^3 = ^5.
#[test]
fn xor_composition_refuses_a_carrying_walk() {
    let (a, b) = (layout("16:^1"), layout("4:3"));
    let pycute = layout("4:^3");
    assert_eq!(a.at(at(&b, 3)), xor(9));
    assert_eq!(pycute.at(3), xor(5));
    assert!(carries(&a.compose(&Tiler::from(b))));
    assert_eq!(a.compose(&Tiler::from(layout("4:2"))), Ok(layout("4:^2")));
}

// Invariant: a composition whose summed walks share bits of one XOR mode refuses. PyCuTe sums
// the walks and returns (3, 2):(^1, ^3) for both left layouts, on one axis or two.
// Witness: A(B(4)) = A(1 + 3) = ^4, but (3, 2):(^1, ^3) maps 4 to ^1 + ^3 = ^2.
#[test]
fn xor_composition_refuses_carrying_sums() {
    let b = layout("(3, 2):(1, 3)");
    let pycute = layout("(3, 2):(^1, ^3)");
    for a in [layout("16:^1"), layout("(6, 2):(^1, ^6)")] {
        assert_eq!(a.at(at(&b, 4)), xor(4), "{a}");
        assert_eq!(pycute.at(4), xor(2));
        assert!(carries(&a.compose(&Tiler::from(b.clone()))), "{a}");
    }
    let disjoint = layout("(4, 2):(1, 4)");
    assert_eq!(layout("16:^1").compose(&Tiler::from(disjoint)), Ok(layout("(4, 2):(^1, ^4)")));
}

// Invariant: an XOR layout sum whose common domain splits a mode at a non-power of two refuses.
// PyCuTe returns 12:^1 + (3, 4):(^16, ^32) = (3, 4):(^17, ^35).
// Witness: the sum at index 5 is ^5 + ^0 = ^5, but (3, 4):(^17, ^35) maps 5 to ^1.
#[test]
fn xor_layout_addition_refuses_a_carrying_split() {
    let (a, b) = (layout("12:^1"), layout("(3, 4):(^16, ^32)"));
    let pycute = layout("(3, 4):(^17, ^35)");
    assert_eq!(a.at(5).add(&b.at(5)), xor(5));
    assert_eq!(pycute.at(5), xor(1));
    assert!(carries(&a.add(&b)));
    assert_eq!(layout("16:^1").add(&layout("(4, 4):(^2, ^8)")), Ok(layout("16:^3")));
}

// Invariant: a product never concatenates copies whose offsets lie in another codomain.
// PyCuTe returns 4:1 times 2:^4 as (4, 2):(1, ^16).
// Witness: the strides 1 and ^16 of PyCuTe's result have no sum, so no layout holds both.
#[test]
fn xor_product_refuses_mixed_codomains() {
    assert_eq!(Offset::from(1).checked_add(&xor(16)), None);
    let product = layout("4:1").logical_product(&Tiler::from(layout("2:^4")));
    assert!(matches!(product, Err(LayoutError::MixedCodomain { .. })));
    assert!(layout("8:^1").logical_product(&Tiler::from(layout("2:1"))).is_err());
}

// Invariant: the right inverse of an XOR layout satisfies its contract, cut to whole modes.
// PyCuTe returns 3:^1 for (3, 8):(^1, ^5).
// Witness: L(^2) splits ^2 over extent 3 carry-lessly as (^1, ^1) and gives ^4, not ^2.
#[test]
fn xor_right_inverse_is_cut_to_its_contract() {
    let l = layout("(3, 8):(^1, ^5)");
    let pycute = layout("3:^1");
    assert_eq!(l.call_offset(&pycute.at(2)), Ok(xor(4)));
    assert_eq!(l.right_inverse(), layout("1:0"));
    assert_eq!(layout("(8, 8):(^9, ^1)").right_inverse(), layout("(8, 8):(^8, ^9)"));
}

// Invariant: the left inverse of an XOR layout refuses when its contract fails. PyCuTe returns
// 3:1 for (3, 2):(^1, 0).
// Witness: L(R(L(2))) = L(^2), which splits ^2 over extent 3 as (^1, ^1) and gives ^1, not ^2.
#[test]
fn xor_left_inverse_refuses_a_broken_contract() {
    let l = layout("(3, 2):(^1, 0)");
    let pycute = layout("3:1");
    let back = l.call_offset(&pycute.call_offset(&l.at(2)).unwrap()).unwrap();
    assert_eq!((l.at(2), back), (xor(2), xor(1)));
    assert!(carries(&l.left_inverse()));
    assert_eq!(layout("(4, 2):(^1, 0)").left_inverse(), Ok(layout("4:1")));
}

// Invariant: the XOR zero is the integer zero, which adds to every codomain. PyCuTe keeps F2(0)
// apart and raises for 16:F0 + 16:1.
// Witness: every offset of 16:^0 + 16:1 is 0 + i = i.
#[test]
fn the_xor_zero_adds_to_every_codomain() {
    let sum = layout("16:^0").add(&layout("16:1")).unwrap();
    assert_eq!(sum, layout("16:1"));
    assert!(!layout("16:^0").stride().is_xor());
}

// Invariant: XOR strides order by their bits, as PyCuTe's F2 does, so the operations that only
// sort strides or pick out zero strides treat them like integers.
// Witness: PyCuTe's make_layout_like and nullspace of XOR layouts.
#[test]
fn xor_strides_order_by_their_bits() {
    assert_eq!(layout("(4, 8):(^8, ^1)").compact_like(), layout("(4, 8):(8, 1)"));
    assert_eq!(layout("(2, 3, 4):(0, ^36, ^4)").compact_like(), layout("(2, 3, 4):(0, 4, 1)"));
    assert_eq!(layout("(4, 8):(^1, 0)").nullspace(), layout("8:4"));
}
