// Copyright © 2026 Dedalus Labs, Inc.

//! Codomain elements: integers and arithmetic tuples (Cecka, Section 2.2.1).
//!
//! A memory layout maps into Z. A coordinate layout maps into Z^n, where each stride is a sum of
//! scaled basis elements `d * E(i)`, as for TMA coordinates or tensor-memory (lane, column)
//! addresses. Both are one type here: a scalar is the rank-0 case of an arithmetic tuple.

use crate::int::Int;
use crate::param::Param;
use crate::tuple::Tuple;
use std::cmp::Ordering;
use std::fmt;

/// An integer, or an arithmetic tuple of integers.
///
/// The value is canonical: trailing zero components are trimmed, and an all-zero tuple is the
/// integer 0. Equal offsets are therefore structurally equal, and `0` adds to any codomain.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Offset(Tuple<Int>);

const ZERO: Tuple<Int> = Tuple::Leaf(Int::Static(0));

impl Offset {
    pub fn zero() -> Self {
        Offset(ZERO)
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

    pub fn from_tuple(tuple: Tuple<Int>) -> Self {
        Offset(canonical(tuple))
    }

    pub fn as_tuple(&self) -> &Tuple<Int> {
        &self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0 == ZERO
    }

    pub fn is_static(&self) -> bool {
        self.0.leaves().iter().all(|value| value.is_static())
    }

    /// Returns the integer this offset is, if it is one.
    pub fn as_int(&self) -> Option<&Int> {
        match &self.0 {
            Tuple::Leaf(value) => Some(value),
            Tuple::Node(_) => None,
        }
    }

    /// Returns the componentwise sum.
    ///
    /// # Panics
    ///
    /// Panics when one offset is a nonzero integer and the other an arithmetic tuple. Every
    /// [`crate::Stride`] holds offsets of one codomain, so the algebra never adds such a pair.
    #[must_use]
    pub fn add(&self, other: &Offset) -> Offset {
        self.checked_add(other)
            .unwrap_or_else(|| panic!("{self} and {other} lie in different codomains"))
    }

    /// Returns the componentwise sum, or `None` when the offsets lie in different codomains.
    pub fn checked_add(&self, other: &Offset) -> Option<Offset> {
        add(&self.0, &other.0).map(Offset::from_tuple)
    }

    #[must_use]
    pub fn sub(&self, other: &Offset) -> Offset {
        self.add(&other.scale(&Int::Static(-1)))
    }

    #[must_use]
    pub fn scale(&self, factor: &Int) -> Offset {
        Offset::from_tuple(self.0.map(&mut |value| value * factor))
    }

    /// Returns the nonzero `(coefficient, path)` terms of this offset as a sum of `c * E(path)`.
    /// Zero is the single term `(0, [])`, as in PyCuTe's `basis_repr`.
    pub fn terms(&self) -> Vec<(Int, Vec<usize>)> {
        let mut terms = Vec::new();
        collect_terms(&self.0, &mut Vec::new(), &mut terms);
        if terms.is_empty() {
            terms.push((Int::Static(0), Vec::new()));
        }
        terms
    }

    /// Returns `(c, path)` when this offset is the single scaled basis element `c * E(path)`.
    pub fn as_basis(&self) -> Option<(Int, Vec<usize>)> {
        let mut terms = self.terms();
        (terms.len() == 1).then(|| terms.pop().expect("one term"))
    }

    /// Compares offsets colexicographically, last component first, if the order is known.
    pub fn compare(&self, other: &Offset) -> Option<Ordering> {
        colex(&self.0, &other.0)
    }

    #[must_use]
    pub fn eval(&self, value: &impl Fn(&Param) -> i64) -> Offset {
        Offset::from_tuple(self.0.map(&mut |v| Int::Static(v.eval(value))))
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
        Offset(Tuple::Leaf(value))
    }
}

impl From<i64> for Offset {
    fn from(value: i64) -> Self {
        Offset(Tuple::Leaf(Int::Static(value)))
    }
}

/// Prints PyCuTe's notation: `5` for an integer, `5@2@1` for `5 * E(1, 2)`, and the components
/// of any other tuple.
impl fmt::Display for Offset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.0, self.as_basis()) {
            (Tuple::Leaf(value), _) => write!(f, "{value}"),
            (_, Some((value, path))) => {
                write!(f, "{value}")?;
                path.iter().rev().try_for_each(|mode| write!(f, "@{mode}"))
            }
            (tuple, None) => write!(f, "{tuple}"),
        }
    }
}

impl fmt::Debug for Offset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
