// Copyright © 2026 Dedalus Labs, Inc.

//! What the parameter facts prove about a predicate.
//!
//! A static extent answers every question the algebra asks. A dynamic extent answers only what
//! its polynomial and its parameters' divisors prove. [`Truth`] keeps the third answer, "the
//! facts cannot tell", visible at every call site, so each site states its policy:
//!
//! - An optional rewrite, such as merging two modes, runs only when [`Truth::is_proven`]. Skipping
//!   it leaves a correct layout that is less simplified.
//! - A precondition, such as shape divisibility, calls [`Truth::require`]. It refuses with
//!   [`Verdict::Violated`] or [`Verdict::Unproven`] and never guesses.
//!
//! ```
//! use tiki_cute::{Condition, Int, Param, Truth};
//!
//! let n = Int::from(Param::multiple_of("N", 8)?);
//! assert_eq!(n.is_multiple_of(&Int::from(4)), Truth::Proven);
//! assert_eq!(n.is_multiple_of(&Int::from(16)), Truth::Open);
//! assert!(n.is_multiple_of(&Int::from(16)).require("tile", Condition::ShapeDivisibility).is_err());
//! # Ok::<(), tiki_cute::LayoutError>(())
//! ```

use crate::error::{Condition, LayoutError, Verdict};

/// The answer to a predicate over every launch the parameter facts admit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truth {
    /// The predicate holds for every admitted launch.
    Proven,
    /// The predicate fails for every admitted launch.
    Refuted,
    /// The predicate holds for some admitted launches and fails for others, or the facts are too
    /// weak to tell which.
    Open,
}

impl Truth {
    #[must_use]
    pub fn is_proven(self) -> bool {
        self == Truth::Proven
    }

    /// Returns `Ok` for a proven predicate and names the failed `condition` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Condition`] with [`Verdict::Violated`] for a refuted predicate and
    /// [`Verdict::Unproven`] for an open one.
    pub fn require(self, operation: &'static str, condition: Condition) -> Result<(), LayoutError> {
        let verdict = match self {
            Truth::Proven => return Ok(()),
            Truth::Refuted => Verdict::Violated,
            Truth::Open => Verdict::Unproven,
        };
        Err(LayoutError::Condition { operation, condition, verdict })
    }

    /// Returns the truth of "`self` or `other`".
    #[must_use]
    pub fn or(self, other: Truth) -> Truth {
        match (self, other) {
            (Truth::Proven, _) | (_, Truth::Proven) => Truth::Proven,
            (Truth::Refuted, Truth::Refuted) => Truth::Refuted,
            _ => Truth::Open,
        }
    }

    /// Returns the truth of "`self` and `other`".
    #[must_use]
    pub fn and(self, other: Truth) -> Truth {
        match (self, other) {
            (Truth::Refuted, _) | (_, Truth::Refuted) => Truth::Refuted,
            (Truth::Proven, Truth::Proven) => Truth::Proven,
            _ => Truth::Open,
        }
    }
}

impl From<bool> for Truth {
    fn from(holds: bool) -> Self {
        if holds { Truth::Proven } else { Truth::Refuted }
    }
}
