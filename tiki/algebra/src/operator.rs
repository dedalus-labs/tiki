// Copyright © 2026 Dedalus Labs, Inc.

//! Operations as marker types, so one type can carry several structures.

/// A binary operation, named by a zero-sized marker type.
pub trait Operator: Copy + 'static {}

/// Addition: `+` on numbers, XOR in [`crate::F2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Add;

/// Multiplication: `×` on numbers, AND in [`crate::F2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mul;

/// The larger of two values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Max;

/// The smaller of two values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Min;

/// `log(eᵃ + eᵇ)`, the addition of the log semiring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogSumExp;

/// Logical or, the addition of the boolean semiring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Or;

/// Logical and, the multiplication of the boolean semiring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct And;

impl Operator for Add {}
impl Operator for Mul {}
impl Operator for Max {}
impl Operator for Min {}
impl Operator for LogSumExp {}
impl Operator for Or {}
impl Operator for And {}
