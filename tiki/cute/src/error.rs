// Copyright © 2026 Dedalus Labs, Inc.

//! Invalid layouts, tensors, and swizzles are rejected before they produce addresses. Each error
//! names the rule it broke and the values that broke it.

use std::fmt;

/// A precondition of a layout operation (Cecka, Section 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    /// Composition: each stride of the right layout divides, or is divided by, the left extents.
    StrideDivisibility,
    /// Composition: each extent of the right layout divides the left extents it spans.
    ShapeDivisibility,
    /// Composition: the modes of a hierarchical right layout have segregated images, so no sum
    /// of their offsets carries across a mode of the left layout (Equation 23).
    SegregatedImages,
    /// Complement and left inverse: no two coordinates reach the same offset.
    Injective,
    /// Left inverse: each stride is a multiple of the extents below it.
    OrderedChain,
    /// Recast: each stride and the scale factor divide one another.
    RecastDivisibility,
    /// Layout addition: both layouts have the same size.
    SizeMatch,
    /// Layout addition: both layouts refine one common domain, computed from static extents.
    CommonDomain,
}

/// Why a precondition failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The condition fails for every launch the parameter facts admit.
    Violated,
    /// The condition depends on a launch parameter the facts do not pin down. A stronger fact,
    /// such as a divisor, may prove it.
    Unproven,
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Condition::StrideDivisibility => "stride divisibility",
            Condition::ShapeDivisibility => "shape divisibility",
            Condition::SegregatedImages => "segregated images",
            Condition::Injective => "injectivity",
            Condition::OrderedChain => "ordered stride chain",
            Condition::RecastDivisibility => "recast divisibility",
            Condition::SizeMatch => "equal size",
            Condition::CommonDomain => "common domain",
        };
        write!(f, "{name}")
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Violated => write!(f, "is violated"),
            Verdict::Unproven => write!(f, "cannot be proven from the parameter facts"),
        }
    }
}

/// Every way a layout, tensor or swizzle operation can refuse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    /// A precondition of a layout operation failed or could not be proven.
    #[error("{operation}: the {condition} condition {verdict}")]
    Condition {
        /// The operation that checked the condition, such as `"composition"`.
        operation: &'static str,
        /// The precondition that failed.
        condition: Condition,
        /// Whether it fails for every launch or only cannot be proven.
        verdict: Verdict,
    },
    /// A stride that is not weakly congruent to its shape.
    #[error("stride {stride} is not congruent to shape {shape}")]
    Incongruent {
        /// The shape, as printed.
        shape: String,
        /// The stride, as printed.
        stride: String,
    },
    /// An extent the parameter facts do not prove positive.
    #[error("extent {extent} is not provably positive")]
    Extent {
        /// The extent, as printed.
        extent: String,
    },
    /// A stride with both nonzero integers and arithmetic tuples, which have no common sum.
    #[error("strides {stride} mix integers with arithmetic tuples")]
    MixedCodomain {
        /// The stride, as printed.
        stride: String,
    },
    /// A coordinate that is not weakly congruent to the shape it indexes.
    #[error("coordinate {coord} does not match shape {shape}")]
    Coordinate {
        /// The coordinate, as printed.
        coord: String,
        /// The shape, as printed.
        shape: String,
    },
    /// A tiler with more modes than the layout it applies to.
    #[error("{operation}: tiler has {tiler} modes but the layout has only {layout}")]
    TilerRank {
        /// The operation the tiler was passed to.
        operation: &'static str,
        /// Number of top-level modes of the tiler.
        tiler: usize,
        /// Number of top-level modes of the layout.
        layout: usize,
    },
    /// A stride with several basis terms where the operation needs one, such as `1@0 + 1@1`.
    #[error("{operation}: stride {stride} is not a scaled basis element")]
    NotBasis {
        /// The operation that needed a single basis term.
        operation: &'static str,
        /// The stride, as printed.
        stride: String,
    },
    /// Operands whose structures do not fit together, described in `detail`.
    #[error("{operation}: {detail}")]
    Mismatch {
        /// The operation that received the operands.
        operation: &'static str,
        /// What does not fit, with the values involved.
        detail: String,
    },
    /// A launch parameter declared with a divisor below 1.
    #[error("parameter {name} needs a positive divisor, got {divisor}")]
    Divisor {
        /// The parameter's name.
        name: String,
        /// The rejected divisor.
        divisor: i64,
    },
    /// Text that is not CuTe notation.
    #[error("cannot parse {text:?} at byte {at}: expected {expected}")]
    Parse {
        /// The whole input.
        text: String,
        /// Byte offset where parsing stopped.
        at: usize,
        /// What the parser expected there.
        expected: &'static str,
    },
    /// A tensor with a zero-byte element.
    #[error("element size must be nonzero")]
    ZeroElementSize,
    /// A tensor layout the byte check cannot evaluate: a dynamic extent or an arithmetic-tuple
    /// stride.
    #[error("tensor layout {layout} must have static extents and integer strides")]
    TensorLayout {
        /// The layout, in CuTe notation.
        layout: String,
    },
    /// A byte address that does not fit in 64 bits.
    #[error("address arithmetic overflows a 64-bit offset")]
    Overflow,
    /// A tensor with an element outside its storage.
    #[error("tensor reaches bytes {lower}..{upper} outside its bounds {start}..{end}")]
    OutsideBounds {
        /// First byte the tensor can read.
        lower: i64,
        /// One past the last byte the tensor can read.
        upper: u64,
        /// First byte the tensor may read.
        start: u64,
        /// One past the last byte the tensor may read.
        end: u64,
    },
    /// A recast the tensor's offset or strides cannot support.
    #[error("cannot recast {from}-byte elements as {to}-byte elements: {reason}")]
    Recast {
        /// Bytes per element before the recast.
        from: u64,
        /// Bytes per element after the recast.
        to: u64,
        /// Which requirement failed.
        reason: &'static str,
    },
    /// A swizzle field width outside `0..=63`.
    #[error("swizzle bits must be in 0..=63, got {bits}")]
    Bits {
        /// Requested width before narrowing to the supported index representation.
        bits: i64,
    },
    /// A swizzle base outside `0..=63`.
    #[error("swizzle base must be in 0..=63, got {base}")]
    Base {
        /// Requested number of untouched low bits.
        base: i64,
    },
    /// Swizzle fields that overlap, `abs(shift) < bits`.
    #[error(
        "swizzle fields overlap: abs(shift) must be at least bits, got bits={bits}, shift={shift}"
    )]
    Overlap {
        /// Width of each source and destination field.
        bits: i64,
        /// Signed distance between the fields.
        shift: i64,
    },
    /// Swizzle fields that reach past bit 62.
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
    /// A negative index passed to a swizzle.
    #[error("swizzle index must be nonnegative, got {index}")]
    NegativeIndex {
        /// Rejected element offset, before any bit operation.
        index: i64,
    },
}
