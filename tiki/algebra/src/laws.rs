// Copyright © 2026 Dedalus Labs, Inc.

//! Checks of each structure's laws on given values.
//!
//! A trait asserts its laws and cannot prove them. These functions check them on values, so a
//! test can run them on random inputs for every implementation. Each returns whether the law
//! holds for its arguments under `equal`, which the float tests pass as a tolerance.

use crate::operator::Operator;
use crate::structure::{CommutativeMonoid, Group, Magma, Monoid, Semiring};

/// Returns whether `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`.
pub fn is_associative<O: Operator, T: Magma<O>>(
    a: &T,
    b: &T,
    c: &T,
    equal: impl Fn(&T, &T) -> bool,
) -> bool {
    let left = Magma::<O>::operate(&Magma::<O>::operate(a, b), c);
    let right = Magma::<O>::operate(a, &Magma::<O>::operate(b, c));
    equal(&left, &right)
}

/// Returns whether `e ⊕ a = a ⊕ e = a`.
pub fn is_identity<O: Operator, T: Monoid<O>>(a: &T, equal: impl Fn(&T, &T) -> bool) -> bool {
    let e = T::identity();
    equal(&Magma::<O>::operate(&e, a), a) && equal(&Magma::<O>::operate(a, &e), a)
}

/// Returns whether `a ⊕ b = b ⊕ a`.
pub fn is_commutative<O: Operator, T: CommutativeMonoid<O>>(
    a: &T,
    b: &T,
    equal: impl Fn(&T, &T) -> bool,
) -> bool {
    equal(&Magma::<O>::operate(a, b), &Magma::<O>::operate(b, a))
}

/// Returns whether `a ⊕ a⁻¹ = e`.
pub fn is_inverse<O: Operator, T: Group<O>>(a: &T, equal: impl Fn(&T, &T) -> bool) -> bool {
    equal(&Magma::<O>::operate(a, &a.inverse()), &T::identity())
}

/// Returns whether `⊗` distributes over `⊕` on both sides and the `⊕` identity annihilates.
pub fn is_distributive<A: Operator, M: Operator, T: Semiring<A, M>>(
    a: &T,
    b: &T,
    c: &T,
    equal: impl Fn(&T, &T) -> bool,
) -> bool {
    let add = |x: &T, y: &T| Magma::<A>::operate(x, y);
    let mul = |x: &T, y: &T| Magma::<M>::operate(x, y);
    let zero = <T as Monoid<A>>::identity();
    equal(&mul(a, &add(b, c)), &add(&mul(a, b), &mul(a, c)))
        && equal(&mul(&add(a, b), c), &add(&mul(a, c), &mul(b, c)))
        && equal(&mul(&zero, a), &zero)
}
