// Copyright © 2026 Dedalus Labs, Inc.

//! Printing layouts, in the forms the `tiki.layout` Python API uses.
//!
//! - `Display` prints the keyword constructor, `Layout(shape=(4, 4), stride=(4, 1))`, so every
//!   integer says what it is.
//! - [`Layout::cute`] prints CuTe's `(4, 4):(4, 1)`, for comparison with CUTLASS and PyCuTe.
//! - [`Layout::describe`] prints one row per coordinate leaf with its extent and stride, then
//!   the index formula. With XOR strides, such as `^9`, the formula's `+` is XOR and its `*` the
//!   carry-less product of [`crate::Xor`].
//!
//! ```
//! use tiki_cute::Layout;
//!
//! let tiles: Layout = "((2, 2), (2, 2)):((4, 8), (1, 2))".parse()?;
//! assert_eq!(tiles.to_string(), "Layout(shape=((2, 2), (2, 2)), stride=((4, 8), (1, 2)))");
//! assert_eq!(tiles.cute().to_string(), "((2, 2), (2, 2)):((4, 8), (1, 2))");
//! assert_eq!(
//!     tiles.describe().to_string(),
//!     "coordinate  extent  stride\n\
//!      c[0][0]     2       4\n\
//!      c[0][1]     2       8\n\
//!      c[1][0]     2       1\n\
//!      c[1][1]     2       2\n\
//!      index = 4 * c[0][0] + 8 * c[0][1] + 1 * c[1][0] + 2 * c[1][1]"
//! );
//! # Ok::<(), tiki_cute::LayoutError>(())
//! ```

use crate::layout::Layout;
use std::fmt;

/// Column titles of [`Layout::describe`].
const HEADER: [&str; 3] = ["coordinate", "extent", "stride"];
/// Spaces between the columns of [`Layout::describe`].
const GAP: &str = "  ";

impl Layout {
    /// Returns a value that prints this layout in CuTe's `shape:stride` notation.
    #[must_use]
    pub fn cute(&self) -> Cute<'_> {
        Cute(self)
    }

    /// Returns a value that prints one coordinate leaf per line, then the index formula.
    #[must_use]
    pub fn describe(&self) -> Description<'_> {
        Description(self)
    }
}

/// CuTe's `shape:stride` notation for a layout.
pub struct Cute<'a>(&'a Layout);

/// A table of a layout's coordinate leaves and the formula that combines them.
pub struct Description<'a>(&'a Layout);

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layout(shape={}, stride={})", self.shape(), self.stride())
    }
}

impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl fmt::Display for Cute<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.0.shape(), self.0.stride())
    }
}

impl fmt::Display for Description<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Name each coordinate leaf by its path, `c` for a leaf layout and `c[1][0]` below the
        // top level, and pair it with its extent and stride.
        let layout = self.0;
        let rows: Vec<[String; 3]> = layout
            .shape()
            .as_tuple()
            .leaf_paths()
            .into_iter()
            .zip(layout.shape().extents().into_iter().zip(layout.stride().steps()))
            .map(|(path, (extent, step))| {
                let name = path.iter().fold(String::from("c"), |name, i| format!("{name}[{i}]"));
                [name, extent.to_string(), step.to_string()]
            })
            .collect();

        // Pad every column to its widest cell so the table lines up in a terminal.
        let widths: Vec<usize> = (0..HEADER.len())
            .map(|column| {
                let cells = rows.iter().map(|row| row[column].len());
                cells.chain([HEADER[column].len()]).max().unwrap_or(0)
            })
            .collect();
        let header = HEADER.map(String::from);
        for row in std::iter::once(&header).chain(&rows) {
            let cells = row.iter().zip(&widths).map(|(cell, &width)| format!("{cell:<width$}"));
            writeln!(f, "{}", cells.collect::<Vec<_>>().join(GAP).trim_end())?;
        }

        // The index is the inner product of the coordinate with the strides.
        let terms: Vec<String> =
            rows.iter().map(|[name, _, step]| format!("{step} * {name}")).collect();
        let formula = if terms.is_empty() { String::from("0") } else { terms.join(" + ") };
        write!(f, "index = {formula}")
    }
}
