// Copyright © 2026 Dedalus Labs, Inc.

//! Codomain elements: integers, arithmetic tuples (Cecka, Section 2.2.1) and XOR values.
//!
//! A memory layout maps into Z. A coordinate layout maps into Z^n, where each stride is a sum of
//! scaled basis elements `d * E(i)`, as for TMA coordinates or tensor-memory (lane, column)
//! addresses. Both are one representation here: a scalar is the rank-0 case of an arithmetic
//! tuple. A swizzled layout maps into the XOR codomain of [`crate::Xor`], whose addition is XOR.

use crate::LayoutError;
use crate::int::Int;
use crate::param::Param;
use crate::tuple::Tuple;
use crate::xor::Xor;
use std::cmp::Ordering;
use std::fmt;

/// An integer, an arithmetic tuple of integers, or an XOR value.
///
/// The value is canonical: trailing zero components are trimmed, and an all-zero tuple and the
/// XOR value `^0` are the integer 0. Equal offsets are therefore structurally equal, and `0` adds
/// to any codomain.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Offset(Value);

/// The representation behind an [`Offset`].
#[derive(Clone, PartialEq, Eq, Hash)]
enum Value {
    /// An integer, or an arithmetic tuple of integers. Zero is always this variant.
    Arithmetic(Tuple<Int>),
    /// A nonzero XOR value.
    Xor(Xor),
}

const ZERO: Tuple<Int> = Tuple::Leaf(Int::Static(0));

impl Offset {
    /// Returns the integer 0, which belongs to every codomain.
    pub fn zero() -> Self {
        Offset(Value::Arithmetic(ZERO))
    }

    /// Returns the basis element `E(path)`. The empty path is the integer 1.
    pub fn basis(path: &[usize]) -> Self {
        Self::scaled_basis(Int::Static(1), path)
    }

    /// Returns `value * E(path)`.
    pub fn scaled_basis(value: Int, path: &[usize]) -> Self {
        let lifted = path.iter().rev().fold(Tuple::Leaf(value), |inner, &mode| {
            let mut modes = vec![ZERO; mode];
            modes.push(inner);
            Tuple::Node(modes)
        });
        Offset::from_tuple(lifted)
    }

    /// Returns the integer or arithmetic tuple `tuple`, trimmed to its canonical form.
    pub fn from_tuple(tuple: Tuple<Int>) -> Self {
        Offset(Value::Arithmetic(canonical(tuple)))
    }

    /// Returns the integer or arithmetic tuple this offset is, or `None` for an XOR value.
    pub fn as_tuple(&self) -> Option<&Tuple<Int>> {
        match &self.0 {
            Value::Arithmetic(tuple) => Some(tuple),
            Value::Xor(_) => None,
        }
    }

    /// Returns the XOR value this offset is, if it is a nonzero one.
    pub fn as_xor(&self) -> Option<Xor> {
        match self.0 {
            Value::Xor(value) => Some(value),
            Value::Arithmetic(_) => None,
        }
    }

    /// Returns whether this is 0, the zero of every codomain.
    pub fn is_zero(&self) -> bool {
        self.0 == Value::Arithmetic(ZERO)
    }

    /// Returns whether every component is known at compile time. An XOR value always is.
    pub fn is_static(&self) -> bool {
        match &self.0 {
            Value::Arithmetic(tuple) => tuple.leaves().iter().all(|value| value.is_static()),
            Value::Xor(_) => true,
        }
    }

    /// Returns the integer this offset is, if it is one.
    pub fn as_int(&self) -> Option<&Int> {
        match &self.0 {
            Value::Arithmetic(Tuple::Leaf(value)) => Some(value),
            Value::Arithmetic(Tuple::Node(_)) | Value::Xor(_) => None,
        }
    }

    /// Returns the sum in the offsets' codomain: componentwise for integers and arithmetic
    /// tuples, XOR for XOR values.
    ///
    /// # Panics
    ///
    /// Panics when the offsets lie in different codomains, such as a nonzero integer and an
    /// arithmetic tuple. Every [`crate::Stride`] holds offsets of one codomain, so the algebra
    /// never adds such a pair.
    #[must_use]
    pub fn add(&self, other: &Offset) -> Offset {
        self.checked_add(other)
            .unwrap_or_else(|| panic!("{self} and {other} lie in different codomains"))
    }

    /// Returns the sum, or `None` when the offsets lie in different codomains.
    pub fn checked_add(&self, other: &Offset) -> Option<Offset> {
        match (&self.0, &other.0) {
            (Value::Arithmetic(a), Value::Arithmetic(b)) => add(a, b).map(Offset::from_tuple),
            (Value::Xor(a), Value::Xor(b)) => Some(Offset::from(*a + *b)),
            (Value::Xor(_), Value::Arithmetic(zero)) if *zero == ZERO => Some(self.clone()),
            (Value::Arithmetic(zero), Value::Xor(_)) if *zero == ZERO => Some(other.clone()),
            _ => None,
        }
    }

    /// Returns the difference. XOR is its own inverse, so for XOR values it is the sum.
    ///
    /// # Panics
    ///
    /// Panics when the offsets lie in different codomains, as [`Offset::add`] does.
    #[must_use]
    pub fn sub(&self, other: &Offset) -> Offset {
        let negated = match &other.0 {
            Value::Arithmetic(_) => other.scale(&Int::Static(-1)),
            Value::Xor(_) => other.clone(),
        };
        self.add(&negated)
    }

    /// Returns `factor` times this offset: componentwise for integers and arithmetic tuples, the
    /// carry-less product for an XOR value.
    ///
    /// # Panics
    ///
    /// Panics when this is an XOR value and `factor` is negative or dynamic. A layout with XOR
    /// strides has static extents, and [`Offset::checked_scale`] refuses instead.
    #[must_use]
    pub fn scale(&self, factor: &Int) -> Offset {
        self.checked_scale(factor)
            .unwrap_or_else(|| panic!("{factor} has no carry-less product with {self}"))
    }

    /// Returns `factor` times this offset, or `None` when this is an XOR value and `factor` is
    /// not a static nonnegative integer.
    pub fn checked_scale(&self, factor: &Int) -> Option<Offset> {
        match &self.0 {
            Value::Arithmetic(tuple) => {
                Some(Offset::from_tuple(tuple.map(&mut |value| value * factor)))
            }
            Value::Xor(value) => Some(Offset::from(Xor::try_from(factor).ok()? * *value)),
        }
    }

    /// Returns this stride times the XOR coordinate `coordinate`, as PyCuTe evaluates a layout at
    /// an F2 index. An integer stride acts through its bits, so the product is an XOR value.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::XorOperand`] for a negative or dynamic integer stride, and
    /// [`LayoutError::XorStride`] for an arithmetic tuple, which has no XOR components.
    pub(crate) fn scale_xor(
        &self,
        coordinate: Xor,
        operation: &'static str,
    ) -> Result<Offset, LayoutError> {
        if self.is_zero() || coordinate == Xor::ZERO {
            return Ok(Offset::zero());
        }
        match &self.0 {
            Value::Xor(value) => Ok(Offset::from(*value * coordinate)),
            Value::Arithmetic(Tuple::Leaf(value)) => {
                Ok(Offset::from(Xor::try_from(value)? * coordinate))
            }
            Value::Arithmetic(Tuple::Node(_)) => {
                Err(LayoutError::XorStride { operation, stride: self.to_string() })
            }
        }
    }

    /// Returns the nonzero `(coefficient, path)` terms of this offset as a sum of `c * E(path)`.
    /// Zero is the single term `(0, [])`, as in PyCuTe's `basis_repr`. An XOR value has no
    /// integer coefficient and so no terms.
    pub fn terms(&self) -> Vec<(Int, Vec<usize>)> {
        let Value::Arithmetic(tuple) = &self.0 else { return Vec::new() };
        let mut terms = Vec::new();
        collect_terms(tuple, &mut Vec::new(), &mut terms);
        if terms.is_empty() {
            terms.push((Int::Static(0), Vec::new()));
        }
        terms
    }

    /// Returns `(c, path)` when this offset is the single scaled basis element `c * E(path)`,
    /// never for an XOR value.
    pub fn as_basis(&self) -> Option<(Int, Vec<usize>)> {
        let mut terms = self.terms();
        (terms.len() == 1).then(|| terms.pop().expect("one term"))
    }

    /// Compares offsets colexicographically, last component first, if the order is known. XOR
    /// values compare by their bits read as integers, as PyCuTe orders them.
    pub fn compare(&self, other: &Offset) -> Option<Ordering> {
        match (&self.0, &other.0) {
            (Value::Arithmetic(a), Value::Arithmetic(b)) => colex(a, b),
            (Value::Xor(a), Value::Xor(b)) => Some(a.cmp(b)),
            (Value::Xor(_), Value::Arithmetic(zero)) if *zero == ZERO => Some(Ordering::Greater),
            (Value::Arithmetic(zero), Value::Xor(_)) if *zero == ZERO => Some(Ordering::Less),
            _ => None,
        }
    }

    /// Returns the offset with a concrete value substituted for every launch parameter.
    #[must_use]
    pub fn eval(&self, value: &impl Fn(&Param) -> i64) -> Offset {
        match &self.0 {
            Value::Arithmetic(tuple) => {
                Offset::from_tuple(tuple.map(&mut |v| Int::Static(v.eval(value))))
            }
            Value::Xor(_) => self.clone(),
        }
    }
}

/// Orders static offsets by value, ahead of every dynamic offset.
///
/// The order of two dynamic offsets depends on the launch, so they compare equal and a stable
/// sort keeps them in their original order. PyCuTe sorts the same way. Algorithms that rely on
/// the order prove each step they take from it, so a wrong guess refuses instead of producing a
/// wrong layout.
pub(crate) fn static_first(a: &Offset, b: &Offset) -> Ordering {
    match (a.is_static(), b.is_static()) {
        (true, true) => a.compare(b).expect("static offsets of one codomain are ordered"),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => Ordering::Equal,
    }
}

fn canonical(tuple: Tuple<Int>) -> Tuple<Int> {
    let Tuple::Node(modes) = tuple else { return tuple };
    let mut modes: Vec<_> = modes.into_iter().map(canonical).collect();
    while modes.last() == Some(&ZERO) {
        modes.pop();
    }
    if modes.is_empty() { ZERO } else { Tuple::Node(modes) }
}

/// Adds componentwise, padding the shorter tuple with zeros. A nonzero integer and a tuple lie
/// in different codomains and have no sum.
fn add(a: &Tuple<Int>, b: &Tuple<Int>) -> Option<Tuple<Int>> {
    match (a, b) {
        (Tuple::Leaf(x), Tuple::Leaf(y)) => Some(Tuple::Leaf(x + y)),
        (zero, other) | (other, zero) if *zero == ZERO => Some(other.clone()),
        (Tuple::Node(p), Tuple::Node(q)) => (0..p.len().max(q.len()))
            .map(|i| add(p.get(i).unwrap_or(&ZERO), q.get(i).unwrap_or(&ZERO)))
            .collect::<Option<Vec<_>>>()
            .map(Tuple::Node),
        _ => None,
    }
}

fn colex(a: &Tuple<Int>, b: &Tuple<Int>) -> Option<Ordering> {
    match (a, b) {
        (Tuple::Leaf(x), Tuple::Leaf(y)) => x.compare(y),
        (Tuple::Leaf(_), Tuple::Node(_)) if *a == ZERO => colex(&Tuple::Node(Vec::new()), b),
        (Tuple::Node(_), Tuple::Leaf(_)) if *b == ZERO => colex(a, &Tuple::Node(Vec::new())),
        (Tuple::Node(p), Tuple::Node(q)) => {
            // The last component is the most significant, as in colexicographic order.
            for i in (0..p.len().max(q.len())).rev() {
                let order = colex(p.get(i).unwrap_or(&ZERO), q.get(i).unwrap_or(&ZERO))?;
                if order != Ordering::Equal {
                    return Some(order);
                }
            }
            Some(Ordering::Equal)
        }
        _ => None,
    }
}

fn collect_terms(tuple: &Tuple<Int>, path: &mut Vec<usize>, terms: &mut Vec<(Int, Vec<usize>)>) {
    match tuple {
        Tuple::Leaf(value) if value.is_zero() => {}
        Tuple::Leaf(value) => terms.push((value.clone(), path.clone())),
        Tuple::Node(modes) => {
            for (mode, component) in modes.iter().enumerate() {
                path.push(mode);
                collect_terms(component, path, terms);
                path.pop();
            }
        }
    }
}

impl From<Int> for Offset {
    fn from(value: Int) -> Self {
        Offset(Value::Arithmetic(Tuple::Leaf(value)))
    }
}

impl From<i64> for Offset {
    fn from(value: i64) -> Self {
        Offset(Value::Arithmetic(Tuple::Leaf(Int::Static(value))))
    }
}

/// An XOR value, canonical: `^0` is the integer 0.
impl From<Xor> for Offset {
    fn from(value: Xor) -> Self {
        if value == Xor::ZERO { Offset::zero() } else { Offset(Value::Xor(value)) }
    }
}

/// Prints PyCuTe's notation: `5` for an integer, `5@2@1` for `5 * E(1, 2)`, and the components
/// of any other tuple. An XOR value prints as `^5`, where PyCuTe prints `F5`, so it cannot read
/// as a parameter name.
///
/// Every term of every coefficient carries its own basis suffix, so `(N + 1) * E(0)` prints as
/// `N@0 + 1@0` and `1@0 + 1@1` stays a sum. Both parse back to the same offset, where a tuple
/// such as `(1, 1)` would read as a stride with two modes.
impl fmt::Display for Offset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Value::Xor(value) => return write!(f, "{value}"),
            Value::Arithmetic(Tuple::Leaf(value)) => return write!(f, "{value}"),
            Value::Arithmetic(Tuple::Node(_)) => {}
        }
        let mut first = true;
        for (value, path) in self.terms() {
            let suffix = path
                .iter()
                .rev()
                .fold(String::new(), |suffix, mode| suffix + "@" + &mode.to_string());
            for summand in value.summands() {
                let text = format!("{summand}{suffix}");
                match (first, text.strip_prefix('-')) {
                    (true, _) => write!(f, "{text}")?,
                    (false, Some(positive)) => write!(f, " - {positive}")?,
                    (false, None) => write!(f, " + {text}")?,
                }
                first = false;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Offset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
