// Copyright © 2026 Dedalus Labs, Inc.

//! Edge cases where a layout operation must refuse with a typed error or return the exact
//! layout, never panic, wrap, or silently change meaning.

use tiki_cute::{Bounds, Condition, Int, Layout, LayoutError, Param, Shape, Tensor, Tiler, Tuple};

fn layout(text: &str) -> Layout {
    text.parse().unwrap()
}

// Invariant: a tensor whose byte range does not fit in 128 bits is refused, never wrapped.
// Witness: extent 2^62 + 1 with stride 2^62 and 16-byte elements reaches past 2^128.
#[test]
fn tensor_byte_range_overflow_is_refused() {
    const HUGE_EXTENT: i64 = (1 << 62) + 1;
    const HUGE_STRIDE: i64 = 1 << 62;
    const ELEMENT: u64 = 16;
    let huge = layout(&format!("{HUGE_EXTENT}:{HUGE_STRIDE}"));
    let placed = Tensor::new(huge, 0, ELEMENT, Bounds { start: 0, end: ELEMENT });
    assert_eq!(placed, Err(LayoutError::Overflow));
}

// Invariant: slicing a tensor at a coordinate known only at launch is refused with an error.
// Witness: a dynamic row index into a static 4 by 4 tensor.
#[test]
fn tensor_slice_at_a_dynamic_coordinate_is_refused() {
    let tensor = Tensor::over(layout("(4, 4):(1, 4)"), 4).unwrap();
    let row = Int::from(Param::positive("i"));
    let coord = Tuple::node([Tuple::Leaf(Some(row)), Tuple::Leaf(None)]);
    assert!(matches!(tensor.slice(&coord), Err(LayoutError::TensorLayout { .. })));
}

// Invariant: a recast factor must be positive, and the refusal names recasting.
// Witness: factors 0 and -1 on a byte tensor.
#[test]
fn tensor_recast_needs_a_positive_factor() {
    let bytes = Tensor::over(layout("8:1"), 2).unwrap();
    for factor in [0, -1] {
        assert!(matches!(bytes.recast(factor), Err(LayoutError::Recast { .. })), "{factor}");
    }
}

// Invariant: recasting keeps each mode's direction. A reversed mode stays reversed with its
// extent scaled by the factor, as CuTe's upcast does.
// Witness: CuTe's results for reversed unit and wider strides, and factor 1 as the identity.
#[test]
fn recast_keeps_negative_strides() {
    let cases = [
        ("8:-1", "2", "4:-1"),
        ("8:-2", "2", "8:-1"),
        ("(4, 2):(-2, 1)", "2", "(4, 1):(-1, 1)"),
        ("4:-1", "1", "4:-1"),
        ("(2, 4):(0, 1@1)", "(1, 2)", "(2, 2):(0, 1@1)"),
    ];
    for (input, scale, expected) in cases {
        let scale: Tuple<Int> = scale.parse().unwrap();
        assert_eq!(layout(input).recast(&scale), Ok(layout(expected)), "{input} by {scale}");
    }
}

// Invariant: a recast factor must be positive.
// Witness: factor 0 is refused instead of dividing by zero.
#[test]
fn recast_refuses_a_zero_factor() {
    let refused = layout("4:2").recast(&Tuple::Leaf(Int::from(0)));
    assert!(matches!(refused, Err(LayoutError::Mismatch { .. })), "{refused:?}");
}

// Invariant: composition with strides that cancel still evaluates A at B's offsets.
// Witness: B = (2,1):(2@0,-2@0) sums to zero, yet its first mode walks A's first axis.
#[test]
fn composition_survives_cancelling_strides() {
    let a = layout("((2, 2), (2, 2)):((1, 4), (10, 40))");
    let b = layout("(2, 1):(2@0, -2@0)");
    let r = a.compose(&Tiler::from(b.clone())).unwrap();
    // B's offsets are coordinates of A's two top-level modes, padded with zeros.
    for i in 0..2 {
        let mut coordinate = b.at(i).as_tuple().unwrap().modes().to_vec();
        coordinate.resize(a.rank(), Tuple::Leaf(Int::from(0)));
        assert_eq!(r.at(i), a.call(&Tuple::Node(coordinate)).unwrap(), "index {i}");
    }
}

// Invariant: layouts whose strides lie in incompatible codomains cannot be built.
// Witness: an integer mode beside a basis mode, and E(0) beside E(0, 0).
#[test]
fn incompatible_codomains_are_refused_at_construction() {
    let modes = [layout("2:1"), layout("2:1@0")];
    assert!(matches!(Layout::try_from_modes(modes), Err(LayoutError::MixedCodomain { .. })));
    let nested = "(2, 2):(1@0, 1@0@0)".parse::<Layout>();
    assert!(matches!(nested, Err(LayoutError::MixedCodomain { .. })), "{nested:?}");
}

// Invariant: substituting a value a parameter does not admit is refused.
// Witness: M, a multiple of 8, evaluated at 0 and at 12.
#[test]
fn eval_refuses_inadmissible_values() {
    let rows = Int::from(Param::multiple_of("M", 8).unwrap());
    let column = Layout::from(Shape::try_from(Tuple::Leaf(rows)).unwrap());
    for value in [0, 12] {
        assert!(
            matches!(column.eval(&|_| value), Err(LayoutError::Inadmissible { .. })),
            "{value}"
        );
    }
    assert_eq!(column.eval(&|_| 16), Ok(layout("16:1")));
}

// Invariant: exact division is trusted only for a divisor that is nonzero for every launch.
// Witness: floor(N / 2) is 0 at N = 1, so it does not provably divide itself.
#[test]
fn a_divisor_that_may_be_zero_proves_nothing() {
    let n = Int::from(Param::positive("N"));
    let half = n.div_floor(&Int::from(2));
    assert_ne!(half.is_multiple_of(&half), tiki_cute::Truth::Proven);
    assert_ne!(half.div_floor(&half), Int::from(1));
}

// Invariant: whitespace never changes what a layout means, and any whitespace is accepted.
// Witness: spaced and unspaced spellings parse equal, and a no-break space parses.
#[test]
fn whitespace_is_insignificant() {
    let cases = [
        ("4 : 1", "4:1"),
        ("4:1 ", "4:1"),
        ("(4 , 4):(1, 4)", "(4,4):(1,4)"),
        ("4:1@0 + 2@1", "4:1@0+2@1"),
        ("(N + 1):1", "(N+1):1"),
        ("4:\u{a0}1", "4:1"),
    ];
    for (spaced, packed) in cases {
        assert_eq!(spaced.parse::<Layout>(), packed.parse::<Layout>(), "{spaced:?}");
    }
}

// Invariant: CuTe notation printed by the crate parses back to the same layout.
// Witness: multi-term basis strides, a polynomial stride, and a floor quotient.
#[test]
fn printed_layouts_parse_back() {
    let rows = Int::from(Param::positive("M"));
    let tiled = Layout::from(Shape::try_from(Tuple::Leaf(rows)).unwrap())
        .logical_divide(&Tiler::try_from(128).unwrap())
        .unwrap();
    let texts = [
        "4:1@0+1@1",
        "(N+1, 2):(1, N+1)",
        "4:N@0+1@0",
        "3:2*N@1+N@0",
        "(2, 2):(1, N - 1)",
        "4:floor((N + 127)/128)",
    ];
    let mut layouts: Vec<Layout> = texts.iter().map(|text| layout(text)).collect();
    layouts.push(tiled);
    for layout in layouts {
        let printed = layout.cute().to_string();
        assert_eq!(printed.parse::<Layout>(), Ok(layout.clone()), "{printed}");
    }
}

// Invariant: every refusal names the operation and rule the caller invoked.
// Witness: an unordered complement and a composition whose tiler addresses a missing axis.
#[test]
fn refusals_name_the_right_rule() {
    let unordered = layout("(3, 2):(2, 3)").complement();
    assert!(matches!(
        unordered,
        Err(LayoutError::Condition { condition: Condition::OrderedChain, .. })
    ));
    let missing_axis = layout("4:1").compose(&Tiler::from(layout("2:1@1")));
    assert!(
        matches!(missing_axis, Err(LayoutError::TilerRank { operation: "composition", .. })),
        "{missing_axis:?}"
    );
}
