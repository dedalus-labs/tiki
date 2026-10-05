// Copyright © 2026 Dedalus Labs, Inc.

//! Why a domain split fails its proof.

use tiki_cute::{Int, Layout, Offset, Truth};

/// Why [`Partition::new`](crate::Partition::new) refuses a split. Each variant
/// carries the layouts and values that break the rule.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PartitionError {
    /// The thread-value layout needs exactly two modes, the thread and then the
    /// value within it, each of static size.
    #[error("thread-value layout {} needs a static thread mode and value mode", thread_values.cute())]
    ThreadShape {
        /// The layout without that shape.
        thread_values: Layout,
    },

    /// A (thread, value) index reaches an offset outside the tile.
    #[error("thread-value layout {} sends index {index} to {offset}, outside the tile", thread_values.cute())]
    OutsideTile {
        /// The layout that leaves the tile.
        thread_values: Layout,
        /// The first index that does.
        index: i64,
        /// Where that index lands.
        offset: Offset,
    },

    /// Two (thread, value) indices reach one tile element, so two threads would
    /// write it.
    #[error("thread-value layout {} sends indices {first} and {second} to element {element}", thread_values.cute())]
    Overlap {
        /// The layout that overlaps.
        thread_values: Layout,
        /// The index that reached the element first.
        first: i64,
        /// The index that reached it again.
        second: i64,
        /// The element both reach.
        element: i64,
    },

    /// The tile does not divide the domain for every launch the facts admit.
    /// [`Truth::Refuted`] means it never divides; [`Truth::Open`] means the
    /// facts are too weak, and a divisibility fact on a launch parameter would
    /// settle it.
    #[error("tile {tile} does not divide {extent} for every launch ({truth:?})")]
    Undivided {
        /// Elements in the tile.
        tile: i64,
        /// The domain's size.
        extent: Int,
        /// Why the division is not proven.
        truth: Truth,
    },
}
