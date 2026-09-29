// Copyright © 2026 Dedalus Labs, Inc.

//! What a proven partition guarantees, and why a partition fails its proof.

use tiki_cute::{Int, Layout, Offset, Truth};

/// Evidence that a [`Partition`](crate::Partition) is safe to lower.
///
/// Only [`Partition::prove`](crate::Partition::prove) makes one, so holding a
/// `Proof` means the grid it describes reaches every element of the domain
/// exactly once, for every launch the domain's facts admit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    /// Blocks in the grid: the domain's size over the tile, exact for every
    /// admitted launch because the tile divides the domain.
    blocks: Int,
    /// Threads in each block.
    threads: i64,
    /// Elements each thread owns in its block's tile.
    values: i64,
}

impl Proof {
    pub(crate) fn new(blocks: Int, threads: i64, values: i64) -> Self {
        Proof { blocks, threads, values }
    }

    /// Blocks to launch, as a function of the launch parameters.
    pub fn blocks(&self) -> &Int {
        &self.blocks
    }

    /// Threads to launch in each block.
    pub fn threads(&self) -> i64 {
        self.threads
    }

    /// Elements each thread computes.
    pub fn values(&self) -> i64 {
        self.values
    }
}

/// Why a partition fails its proof. Each variant carries the layouts and values
/// that break the rule.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ProofError {
    /// The thread-value layout needs exactly two modes, the thread and then the
    /// value within it, each of static size.
    #[error("thread-value layout {} needs a static thread mode and value mode", threads.cute())]
    ThreadShape {
        /// The layout without that shape.
        threads: Layout,
    },

    /// The thread-value layout's size differs from the tile, so some tile
    /// element has no owner.
    #[error("thread-value layout {} covers {size} elements; the tile has {tile}", threads.cute())]
    TileSize {
        /// The layout whose size is wrong.
        threads: Layout,
        /// Its size.
        size: Int,
        /// Elements in the tile.
        tile: i64,
    },

    /// A (thread, value) index reaches an offset outside the tile.
    #[error("thread-value layout {} sends index {index} to {offset}, outside the tile", threads.cute())]
    OutsideTile {
        /// The layout that leaves the tile.
        threads: Layout,
        /// The first index that does.
        index: i64,
        /// Where that index lands.
        offset: Offset,
    },

    /// Two (thread, value) indices reach one tile element, so two threads would
    /// write it.
    #[error("thread-value layout {} sends indices {first} and {second} to element {element}", threads.cute())]
    Overlap {
        /// The layout that overlaps.
        threads: Layout,
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
