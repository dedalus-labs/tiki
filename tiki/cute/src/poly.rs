// Copyright © 2026 Dedalus Labs, Inc.

//! Integer polynomials over launch parameters, the value of a dynamic extent or stride.
//!
//! A polynomial is a sum of monomials with nonzero integer coefficients, each monomial a sorted
//! product of atoms. An atom is a parameter or an opaque quotient `floor(p / q)` that no exact
//! division could remove. The representation is canonical, so equal polynomials compare equal
//! structurally and the layout algebra needs no separate simplifier.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::param::Param;

pub(crate) const OVERFLOW: &str = "layout arithmetic overflows i64";

/// An irreducible factor of a monomial.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    /// A launch parameter.
    Param(Param),
    /// `floor(dividend / divisor)` where no exact division exists, such as `ceil(N / 128)`
    /// written as `floor((N + 127) / 128)`.
    Floor(Arc<Poly>, Arc<Poly>),
}

/// A sum of monomials with nonzero coefficients, keyed by the monomial's sorted atoms. The
/// constant term has the empty monomial.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Poly(BTreeMap<Vec<Atom>, i64>);

impl Poly {
    pub fn constant(value: i64) -> Self {
        Self::term(value, Vec::new())
    }

    pub fn atom(atom: Atom) -> Self {
        Self::term(1, vec![atom])
    }

    fn term(coefficient: i64, monomial: Vec<Atom>) -> Self {
        let mut poly = Self::default();
        poly.accumulate(monomial, coefficient);
        poly
    }

    fn accumulate(&mut self, monomial: Vec<Atom>, coefficient: i64) {
        let entry = self.0.entry(monomial).or_insert(0);
        *entry = entry.checked_add(coefficient).expect(OVERFLOW);
        self.0.retain(|_, c| *c != 0);
    }

    pub fn as_constant(&self) -> Option<i64> {
        match self.0.iter().next() {
            None => Some(0),
            Some((monomial, &c)) if monomial.is_empty() && self.0.len() == 1 => Some(c),
            Some(_) => None,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut sum = self.clone();
        for (monomial, &c) in &other.0 {
            sum.accumulate(monomial.clone(), c);
        }
        sum
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut product = Self::default();
        for (a, &ca) in &self.0 {
            for (b, &cb) in &other.0 {
                let mut monomial: Vec<Atom> = a.iter().chain(b).cloned().collect();
                monomial.sort();
                product.accumulate(monomial, ca.checked_mul(cb).expect(OVERFLOW));
            }
        }
        product
    }

    /// `self / divisor` when it is a polynomial: the divisor is one term whose atoms and
    /// coefficient divide every term of `self`.
    pub fn exact_div(&self, divisor: &Self) -> Option<Self> {
        let [(atoms, &c)] = divisor.0.iter().collect::<Vec<_>>()[..] else { return None };
        let mut quotient = Self::default();
        for (monomial, &coefficient) in &self.0 {
            let rest = remove_all(monomial, atoms)?;
            (coefficient % c == 0).then_some(())?;
            quotient.accumulate(rest, coefficient / c);
        }
        Some(quotient)
    }

    /// The largest integer every value of `self` is known to be a multiple of, from the
    /// coefficients and each parameter's divisor. Zero for the zero polynomial.
    pub fn known_multiple(&self) -> i64 {
        self.0.iter().fold(0, |g, (monomial, &c)| {
            let magnitude = c.checked_abs().expect(OVERFLOW);
            let term = monomial.iter().fold(magnitude, |m, atom| match atom {
                Atom::Param(p) => m.checked_mul(p.divisor()).expect(OVERFLOW),
                Atom::Floor(..) => m,
            });
            gcd(g, term)
        })
    }

    /// Splits the terms into those `divisor` provably divides, from their coefficients and the
    /// parameters' divisors, and the rest.
    pub fn split_multiples(&self, divisor: i64) -> (Poly, Poly) {
        let (mut multiples, mut rest) = (Poly::default(), Poly::default());
        for (monomial, &c) in &self.0 {
            let term = Poly::term(c, monomial.clone());
            let part =
                if term.known_multiple() % divisor == 0 { &mut multiples } else { &mut rest };
            part.accumulate(monomial.clone(), c);
        }
        (multiples, rest)
    }

    /// Returns bounds `[low, high]` over every assignment of the parameters, `None` where
    /// unbounded. A parameter is at least its divisor. A quotient is bounded below as
    /// [`Atom::least`] describes.
    pub fn bounds(&self) -> (Option<i128>, Option<i128>) {
        let (mut low, mut high) = (Some(0i128), Some(0i128));
        for (monomial, &c) in &self.0 {
            let least = monomial.iter().try_fold(1i128, |m, atom| Some(m * atom.least()?));
            let term = least.and_then(|m| m.checked_mul(c.into()));
            let (term_low, term_high) = match (term, c > 0 || monomial.is_empty()) {
                (Some(t), _) if monomial.is_empty() => (Some(t), Some(t)),
                (Some(t), true) => (Some(t), None),
                (Some(t), false) => (None, Some(t)),
                (None, _) => (None, None),
            };
            low = low.zip(term_low).and_then(|(a, b)| a.checked_add(b));
            high = high.zip(term_high).and_then(|(a, b)| a.checked_add(b));
        }
        (low, high)
    }

    pub fn eval(&self, value: &impl Fn(&Param) -> i64) -> i64 {
        self.0.iter().fold(0i64, |sum, (monomial, &c)| {
            let term =
                monomial.iter().fold(c, |m, atom| m.checked_mul(atom.eval(value)).expect(OVERFLOW));
            sum.checked_add(term).expect(OVERFLOW)
        })
    }
}

impl Atom {
    fn least(&self) -> Option<i128> {
        match self {
            Atom::Param(p) => Some(p.divisor().into()),
            // A constant divisor gives the exact floor of the dividend's least value, which
            // proves `ceil(N / c) >= 1` for a positive `N`. Any other positive divisor only
            // proves the quotient is nonnegative.
            Atom::Floor(p, q) => {
                let low = p.bounds().0.filter(|&low| low >= 0)?;
                match q.as_constant() {
                    Some(c) if c >= 1 => Some(low / i128::from(c)),
                    _ => q.bounds().0.is_some_and(|low| low >= 1).then_some(0),
                }
            }
        }
    }

    fn eval(&self, value: &impl Fn(&Param) -> i64) -> i64 {
        match self {
            Atom::Param(p) => value(p),
            Atom::Floor(p, q) => floor_div(p.eval(value), q.eval(value)),
        }
    }
}

/// `monomial` with one copy of each atom in `atoms` removed, if it contains them all.
fn remove_all(monomial: &[Atom], atoms: &[Atom]) -> Option<Vec<Atom>> {
    let mut rest = monomial.to_vec();
    for atom in atoms {
        let at = rest.iter().position(|a| a == atom)?;
        rest.remove(at);
    }
    Some(rest)
}

/// Division rounding toward negative infinity, as Python's `//`.
pub(crate) fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if a % b != 0 && (a < 0) != (b < 0) { q - 1 } else { q }
}

pub(crate) fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "0");
        }
        for (i, (monomial, &c)) in self.0.iter().rev().enumerate() {
            let sign = match (i, c < 0) {
                (0, false) => "",
                (0, true) => "-",
                (_, false) => " + ",
                (_, true) => " - ",
            };
            let mut factors: Vec<String> = monomial.iter().map(ToString::to_string).collect();
            if c.unsigned_abs() != 1 || factors.is_empty() {
                factors.insert(0, c.unsigned_abs().to_string());
            }
            write!(f, "{sign}{}", factors.join("*"))?;
        }
        Ok(())
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Param(p) => write!(f, "{p}"),
            Atom::Floor(p, q) => write!(f, "floor({}/{})", Operand(p), Operand(q)),
        }
    }
}

/// A polynomial printed as an operand: parenthesized when it has more than one term.
struct Operand<'a>(&'a Poly);

impl fmt::Display for Operand<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.0.len() > 1 { write!(f, "({})", self.0) } else { write!(f, "{}", self.0) }
    }
}
