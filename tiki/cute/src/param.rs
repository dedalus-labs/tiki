// Copyright © 2026 Dedalus Labs, Inc.

//! Launch parameters: the unknowns in a dynamic extent or stride.

use crate::LayoutError;
use std::fmt;
use std::sync::Arc;

/// A positive integer fixed at launch, known at compile time only through its facts.
///
/// Every fact is a promise the launcher checks against the actual argument before the kernel
/// runs. The algebra may rely on it, and a launch that breaks it is refused before any address
/// is computed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Param {
    /// The name shown in layouts and errors. Two parameters with one name and different
    /// divisors are different unknowns.
    name: Arc<str>,
    /// Every admitted value is a positive multiple of this. It is at least 1, so it is also the
    /// least admitted value.
    divisor: i64,
}

impl Param {
    /// Returns a parameter admitting every positive integer.
    #[must_use]
    pub fn positive(name: &str) -> Self {
        Param { name: name.into(), divisor: 1 }
    }

    /// Returns a parameter admitting the positive multiples of `divisor`, such as a leading
    /// dimension that alignment makes a multiple of 8.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Divisor`] when `divisor` is below 1.
    pub fn multiple_of(name: &str, divisor: i64) -> Result<Self, LayoutError> {
        if divisor < 1 {
            return Err(LayoutError::Divisor { name: name.into(), divisor });
        }
        Ok(Param { name: name.into(), divisor })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn divisor(&self) -> i64 {
        self.divisor
    }

    /// Returns whether `value` keeps every promise this parameter makes.
    #[must_use]
    pub fn admits(&self, value: i64) -> bool {
        value >= 1 && value % self.divisor == 0
    }
}

impl fmt::Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}
