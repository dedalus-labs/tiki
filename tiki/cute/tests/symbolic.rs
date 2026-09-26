// Copyright © 2026 Dedalus Labs, Inc.

//! Layouts with dynamic extents: exact results where the facts decide, refusals where they do
//! not, and soundness under every substitution.
//!
//! A symbolic result is only useful if it is right for every launch. The substitution
//! properties evaluate each symbolic result at concrete parameter values and require it to
//! satisfy the same postconditions the static algebra satisfies.

mod common;

use common::*;
use proptest::prelude::*;
use tiki_cute::{
    Condition, Int, Layout, LayoutError, Offset, Param, Shape, Stride, Tiler, Tuple, Verdict,
};

/// Values substituted for each parameter in the soundness properties.
const VALUES: std::ops::RangeInclusive<i64> = 1..=5;

fn layout(text: &str) -> Layout {
    text.parse().unwrap()
}

/// Substitutes `value` for every parameter, whatever its name.
fn at_value(layout: &Layout, value: i64) -> Layout {
    layout.eval(&|_: &Param| value)
}

// Coalescing merges through symbolic strides and extents whenever the merge is an identity of
// polynomials, and leaves unrelated symbols apart. These are PyCuTe's sympy cases.
#[test]
fn coalesce_merges_polynomial_identities() {
    let cases = [
        ("(2, 4):(X, 2*X)", "8:X"),
        ("(2, 3, 4):(X, 2*X, 6*X)", "24:X"),
        ("(2, 4):(X, 3*X)", "(2, 4):(X, 3*X)"),
        ("(4, N):(1, 4)", "4*N:1"),
        ("(N, 4):(1, N)", "4*N:1"),
        ("(2, N):(1, 2)", "2*N:1"),
        ("(N, M):(1, N)", "M*N:1"),
        ("(N+1, 2):(1, N+1)", "2*N+2:1"),
        ("(2, N+1):(1, 2)", "2*N+2:1"),
        ("(N, M):(X, Y)", "(N, M):(X, Y)"),
        ("(1, N):(X, Y)", "N:Y"),
    ];
    for (input, expected) in cases {
        assert_eq!(layout(input).coalesce(), layout(expected), "{input}");
    }
}

// The complement of a layout with symbolic extents and static strides is exact, because the
// strides fix the order of the gaps.
#[test]
fn complement_with_symbolic_extents_is_exact() {
    let cases = [
        ("N:1", "1:N"),
        ("N:X", "(X, 1):(1, N*X)"),
        ("(4, N):(1, 4)", "1:4*N"),
        ("(2, N):(1, 2)", "1:2*N"),
        ("(N, 4):(1, N)", "1:4*N"),
    ];
    for (input, expected) in cases {
        assert_eq!(layout(input).complement().unwrap(), layout(expected), "{input}");
    }
    let cotarget: Shape = "(32, 4, 2, 2, N)".parse().unwrap();
    assert_eq!(layout("256:1").complement_in(&cotarget).unwrap(), layout("2*N:256"));
}

// Dividing a static tile by a symbolic leading extent cannot be proven, so composition refuses
// instead of emitting an unchecked layout.
#[test]
fn composition_refuses_an_unprovable_shape_division() {
    let a = layout("(N, 8):(X, Y)");
    let refused = a.compose(&Tiler::try_from(4).unwrap());
    let expected = LayoutError::Condition {
        operation: "composition",
        condition: Condition::ShapeDivisibility,
        verdict: Verdict::Unproven,
    };
    assert_eq!(refused, Err(expected));
}

// A divisor fact is what lets a tile of a dynamic extent compose. Tiling a padded matrix whose
// row count `M` is a multiple of the tile proves that the tile fits inside the rows, and the
// grid of tiles is exactly `M / 128`.
#[test]
fn a_divisor_fact_proves_the_tile_fits() {
    const TILE: i64 = 128;
    /// A padded leading dimension, so the matrix's two modes never coalesce into one.
    const PITCH: i64 = 4096;
    let aligned = Int::from(Param::multiple_of("M", TILE).unwrap());
    let unaligned = Int::from(Param::positive("M"));
    for (rows, fits) in [(aligned, true), (unaligned, false)] {
        let shape =
            Shape::try_from(Tuple::node([Tuple::Leaf(rows.clone()), Tuple::Leaf(Int::from(8))]))
                .unwrap();
        let stride = Stride::try_from(Tuple::node([
            Tuple::Leaf(Offset::from(1)),
            Tuple::Leaf(Offset::from(PITCH)),
        ]))
        .unwrap();
        let a = Layout::new(shape, stride).unwrap();
        let tile = a.compose(&Tiler::try_from(TILE).unwrap());
        assert_eq!(tile.is_ok(), fits, "{a}: {tile:?}");

        let grid = Layout::from(Shape::try_from(Tuple::Leaf(rows.clone())).unwrap())
            .logical_divide(&Tiler::try_from(TILE).unwrap())
            .unwrap();
        let tiles = grid.shape().extents()[1].clone();
        if fits {
            assert_eq!(tiles, rows.div_floor(&Int::from(TILE)), "{grid}");
        }
    }
}

prop_compose! {
    /// A layout whose leaves draw extents and strides from constants and the parameter `N`.
    fn symbolic_layouts()(leaves in 1..=3usize)(
        extents in prop::collection::vec(prop::sample::select(vec!["1", "2", "3", "4", "N", "2*N"]), leaves),
        strides in prop::collection::vec(prop::sample::select(vec!["0", "1", "2", "4", "N", "2*N", "4*N"]), leaves),
        grouping in 0u8..3,
    ) -> Layout {
        let shape = format!("{}", nest(&extents, grouping)).replace('"', "");
        let stride = format!("{}", nest(&strides, grouping)).replace('"', "");
        layout(&format!("{shape}:{stride}"))
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1024, max_global_rejects: 1 << 20, ..ProptestConfig::default() })]

    // Invariant: a symbolic coalesce has the original's function for every value of N.
    // Witness: substituting N = 1..5 in both gives equal offsets at every index.
    #[test]
    fn symbolic_coalesce_is_sound(l in symbolic_layouts()) {
        let r = l.coalesce();
        for n in VALUES {
            prop_assert_eq!(check_coalesce(&at_value(&l, n), &at_value(&r, n)), Ok(()), "{} at N = {}", l.cute(), n);
        }
    }

    // Invariant: a symbolic composition that succeeds is a composition for every value of N.
    // Witness: substituting N = 1..5, R(i) = A(B(i)) wherever B stays in A's domain.
    #[test]
    fn symbolic_composition_is_sound(a in symbolic_layouts(), b in layouts()) {
        let Ok(r) = a.compose(&Tiler::from(b.clone())) else { return Ok(()) };
        let reach = (0..size(&b)).map(|i| at(&b, i)).max().unwrap_or(0);
        for n in VALUES {
            let a_n = at_value(&a, n);
            if reach < size(&a_n) {
                prop_assert_eq!(check_composition(&a_n, &b, &at_value(&r, n)), Ok(()), "{} o {} at N = {}", a.cute(), b.cute(), n);
            }
        }
    }

    // Invariant: a symbolic complement that succeeds is a complement for every value of N.
    // Witness: substituting N = 1..5 gives an ordered layout disjoint from the input's image.
    #[test]
    fn symbolic_complement_is_sound(l in symbolic_layouts()) {
        let Ok(r) = l.complement() else { return Ok(()) };
        for n in VALUES {
            let l_n = at_value(&l, n);
            prop_assume!(l_n.left_inverse().is_ok());
            prop_assert_eq!(check_complement(&l_n, &at_value(&r, n)), Ok(()), "{} at N = {}", l.cute(), n);
        }
    }

    // Invariant: a symbolic left inverse that succeeds recovers coordinates for every N.
    // Witness: substituting N = 1..5, L(R(L(i))) = L(i) at every index.
    #[test]
    fn symbolic_left_inverse_is_sound(l in symbolic_layouts()) {
        let Ok(r) = l.left_inverse() else { return Ok(()) };
        for n in VALUES {
            prop_assert_eq!(check_left_inverse(&at_value(&l, n), &at_value(&r, n)), Ok(()), "{} at N = {}", l.cute(), n);
        }
    }
}
