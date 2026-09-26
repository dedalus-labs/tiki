// Copyright © 2026 Dedalus Labs, Inc.

//! Every implementation's laws, checked on random values.

use proptest::prelude::*;
use tiki_algebra::laws::{
    is_associative, is_commutative, is_distributive, is_identity, is_inverse,
};
use tiki_algebra::{Add, And, F2, LogSumExp, Max, Min, Mul, Or};

/// Integers small enough that products of three never overflow.
const SMALL: std::ops::RangeInclusive<i64> = -1_000_000..=1_000_000;
/// Relative tolerance for float laws, which hold up to rounding.
const TOLERANCE: f64 = 1e-9;

fn exact<T: PartialEq>(a: &T, b: &T) -> bool {
    a == b
}

fn close(a: &f64, b: &f64) -> bool {
    a == b || (a - b).abs() <= TOLERANCE * a.abs().max(b.abs()).max(1.0)
}

fn bits() -> impl Strategy<Value = F2> {
    any::<bool>().prop_map(F2::from)
}

proptest! {
    // Invariant: the integers are a ring and a monoid under max and min.
    // Witness: every law holds exactly on random integers.
    #[test]
    fn integers_are_a_ring(a in SMALL, b in SMALL, c in SMALL) {
        prop_assert!(is_associative::<Add, _>(&a, &b, &c, exact));
        prop_assert!(is_associative::<Mul, _>(&a, &b, &c, exact));
        prop_assert!(is_associative::<Max, _>(&a, &b, &c, exact));
        prop_assert!(is_associative::<Min, _>(&a, &b, &c, exact));
        prop_assert!(is_identity::<Add, _>(&a, exact) && is_identity::<Mul, _>(&a, exact));
        prop_assert!(is_identity::<Max, _>(&a, exact) && is_identity::<Min, _>(&a, exact));
        prop_assert!(is_commutative::<Add, _>(&a, &b, exact) && is_commutative::<Mul, _>(&a, &b, exact));
        prop_assert!(is_inverse::<Add, _>(&a, exact));
        prop_assert!(is_distributive::<Add, Mul, _>(&a, &b, &c, exact));
    }

    // Invariant: F2 is a field, with XOR as addition.
    // Witness: every law holds on every triple of bits, and one is the only invertible element.
    #[test]
    fn f2_is_a_field(a in bits(), b in bits(), c in bits()) {
        prop_assert!(is_associative::<Add, _>(&a, &b, &c, exact) && is_associative::<Mul, _>(&a, &b, &c, exact));
        prop_assert!(is_identity::<Add, _>(&a, exact) && is_identity::<Mul, _>(&a, exact));
        prop_assert!(is_commutative::<Add, _>(&a, &b, exact) && is_commutative::<Mul, _>(&a, &b, exact));
        prop_assert!(is_inverse::<Add, _>(&a, exact));
        prop_assert!(is_distributive::<Add, Mul, _>(&a, &b, &c, exact));
        use tiki_algebra::Field;
        prop_assert_eq!(a.reciprocal().map(|r| r * a), a.is_one().then_some(F2::ONE));
    }

    // Invariant: floats form the real, tropical, min-plus and log semirings up to rounding.
    // Witness: every law holds within the tolerance on random finite floats.
    #[test]
    fn floats_form_the_semirings(a in -1e3f64..1e3, b in -1e3f64..1e3, c in -1e3f64..1e3) {
        prop_assert!(is_distributive::<Add, Mul, _>(&a, &b, &c, close));
        prop_assert!(is_distributive::<Max, Add, _>(&a, &b, &c, close));
        prop_assert!(is_distributive::<Min, Add, _>(&a, &b, &c, close));
        prop_assert!(is_distributive::<LogSumExp, Add, _>(&a, &b, &c, close));
        prop_assert!(is_associative::<LogSumExp, _>(&a, &b, &c, close));
        prop_assert!(is_identity::<LogSumExp, _>(&a, close) && is_identity::<Max, _>(&a, close));
        prop_assert!(is_commutative::<LogSumExp, _>(&a, &b, close));
    }

    // Invariant: booleans form the boolean semiring.
    // Witness: every law holds on every triple of booleans.
    #[test]
    fn booleans_form_the_boolean_semiring(a: bool, b: bool, c: bool) {
        prop_assert!(is_associative::<Or, _>(&a, &b, &c, exact) && is_associative::<And, _>(&a, &b, &c, exact));
        prop_assert!(is_identity::<Or, _>(&a, exact) && is_identity::<And, _>(&a, exact));
        prop_assert!(is_distributive::<Or, And, _>(&a, &b, &c, exact));
    }
}
