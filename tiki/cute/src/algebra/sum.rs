// Copyright © 2026 Dedalus Labs, Inc.

//! Layout addition: the pointwise sum of two layouts of one size.
//!
//! Composition needs it when a stride of `B` has several basis terms, such as `3@0 + 5@1`. Each
//! term walks its own axis of `A`, and the walks add pointwise. The sum is a layout only on a
//! domain that refines both summands, so both are cut down to their greatest common domain
//! first.
//!
//! An XOR summand is linear on the common domain only where the domain's modes split each of its
//! modes at a power of two. PyCuTe skips that condition, so for `12:^1 + (3, 4):(^16, ^32)` it
//! returns `(3, 4):(^17, ^35)`, which maps index 5 to `^1` where the sum is `^5`.

use crate::LayoutError;
use crate::error::{Condition, Verdict};
use crate::int::Int;
use crate::layout::Layout;
use crate::poly::gcd;
use crate::shape::Shape;
use crate::stride::Stride;
use crate::truth::Truth;
use crate::tuple::Tuple;

const OPERATION: &str = "layout addition";

impl Layout {
    /// Returns the layout `R` with `R(i) = self(i) + other(i)` for every `i` in `0..size`.
    ///
    /// # Errors
    ///
    /// Returns [`Condition::SizeMatch`] when the sizes differ, [`Condition::CommonDomain`] when
    /// no common refinement covers the whole domain or an extent is dynamic, and
    /// [`Condition::CarryFree`] when an XOR sum differs from the pointwise sum at an index.
    pub fn add(&self, other: &Layout) -> Result<Layout, LayoutError> {
        self.size().equals(&other.size()).require(OPERATION, Condition::SizeMatch)?;

        // Refine both summands to one flat domain. Each of its coordinates is a single index into
        // either summand, so the sum of their offsets there is the sum layout's stride.
        let (a, b) = (self.coalesce(), other.coalesce());
        let domain = CommonDomain::between(&static_extents(&a)?, &static_extents(&b)?);
        let covered = Int::Static(domain.extents.iter().product());
        covered.equals(&self.size()).require(OPERATION, Condition::CommonDomain)?;

        let extents = domain.extents.into_iter().map(|s| Tuple::Leaf(Int::Static(s)));
        let mut steps = Vec::new();
        for index in domain.indices {
            let (x, y) = (a.at(index), b.at(index));
            let step = x.checked_add(&y).ok_or_else(|| LayoutError::Mismatch {
                operation: OPERATION,
                detail: format!("{self} and {other} map into different codomains"),
            })?;
            steps.push(Tuple::Leaf(step));
        }
        let shape = Shape::from_derived(Tuple::node(extents));
        let sum = Layout::from_parts(shape, Stride::from_derived(Tuple::Node(steps))).coalesce();

        // An XOR summand's steps add up to its offsets only where the common domain splits its
        // modes at powers of two. Every extent is static here, so check each index.
        if self.stride().is_xor() || other.stride().is_xor() {
            for index in 0..covered.as_static().expect("the common domain is static") {
                let pointwise = self.at(index).checked_add(&other.at(index));
                Truth::from(pointwise == Some(sum.at(index)))
                    .require(OPERATION, Condition::CarryFree)?;
            }
        }
        Ok(sum)
    }
}

/// Returns the leaf extents of a layout when every one is static.
///
/// The common domain is built from gcd and lcm, which have no polynomial form here, so a
/// dynamic extent leaves the refinement open.
fn static_extents(layout: &Layout) -> Result<Vec<i64>, LayoutError> {
    let extents: Option<Vec<i64>> =
        layout.shape().extents().into_iter().map(Int::as_static).collect();
    extents.ok_or(LayoutError::Condition {
        operation: OPERATION,
        condition: Condition::CommonDomain,
        verdict: Verdict::Unproven,
    })
}

/// The coarsest flat layout whose coordinates are coordinates of both summands.
struct CommonDomain {
    /// Extent of each mode of the common domain.
    extents: Vec<i64>,
    /// The 1-D index into either summand where each mode steps, its stride.
    indices: Vec<i64>,
}

impl CommonDomain {
    /// Builds the common domain of two flat shapes. It covers the prefix of the domain on which
    /// the two shapes split alike, the whole domain when they refine one another.
    fn between(a: &[i64], b: &[i64]) -> CommonDomain {
        let (mut a, mut b) = (a.to_vec(), b.to_vec());
        let mut domain = CommonDomain { extents: Vec::new(), indices: Vec::new() };
        // `prefix_a` and `prefix_b` are the sizes of the modes of each shape already consumed.
        let (mut prefix_a, mut prefix_b) = (1i64, 1i64);
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            // Extent-1 modes split nothing.
            if a[i] == 1 {
                i += 1;
                continue;
            }
            if b[j] == 1 {
                j += 1;
                continue;
            }

            // Both shapes reach index `lcm(prefix_a, prefix_b)` by advancing within their current
            // mode. The factor both current modes share from there is one more common mode.
            let lcm = prefix_a / gcd(prefix_a, prefix_b) * prefix_b;
            let (step_a, step_b) = (lcm / prefix_a, lcm / prefix_b);
            if a[i] % step_a == 0 && b[j] % step_b == 0 {
                let shared = gcd(a[i] / step_a, b[j] / step_b);
                if shared != 1 {
                    domain.extents.push(shared);
                    domain.indices.push(lcm);
                    a[i] /= step_a * shared;
                    b[j] /= step_b * shared;
                    prefix_a = lcm * shared;
                    prefix_b = prefix_a;
                    continue;
                }
            }

            // No shared factor is left. Advance whichever shape's mode ends first, or both when
            // the two modes end together.
            let (end_a, end_b) = (prefix_a * a[i], prefix_b * b[j]);
            let (a_ends_first, b_ends_first) = (end_b % end_a == 0, end_a % end_b == 0);
            if a_ends_first || !b_ends_first {
                prefix_a = end_a;
                i += 1;
            }
            if b_ends_first || !a_ends_first {
                prefix_b = end_b;
                j += 1;
            }
        }
        if domain.extents.is_empty() {
            return CommonDomain { extents: vec![1], indices: vec![0] };
        }
        domain
    }
}
