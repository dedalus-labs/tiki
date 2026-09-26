// Copyright © 2026 Dedalus Labs, Inc.

//! Invalid indexing transforms are rejected before they produce addresses.

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    #[error("swizzle bits must be in 0..=63, got {bits}")]
    Bits {
        /// Requested width before narrowing to the supported index representation.
        bits: i64,
    },
    #[error("swizzle base must be in 0..=63, got {base}")]
    Base {
        /// Requested number of untouched low bits.
        base: i64,
    },
    #[error(
        "swizzle fields overlap: abs(shift) must be at least bits, got bits={bits}, shift={shift}"
    )]
    Overlap {
        /// Width of each source and destination field.
        bits: i64,
        /// Signed distance between the fields.
        shift: i64,
    },
    #[error(
        "swizzle fields must fit index bits 0..62, got bits={bits}, base={base}, shift={shift}"
    )]
    Width {
        /// Requested width of each field.
        bits: i64,
        /// Number of low bits below both fields.
        base: i64,
        /// Requested signed field distance.
        shift: i64,
    },
    #[error("swizzle index must be nonnegative, got {index}")]
    NegativeIndex {
        /// Rejected element offset, before any bit operation.
        index: i64,
    },
}
