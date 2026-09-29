// Copyright © 2026 Dedalus Labs, Inc.

//! How an elementwise kernel splits its domain across blocks and threads.
//!
//! Proof:
//!
//!  1. The thread-value layout has a thread mode and a value mode, each of
//!     static size.
//!  2. It reaches each element of the tile exactly once: its size equals the
//!     tile, and every (thread, value) index lands on a distinct element
//!     inside the tile.
//!  3. The tile divides the domain for every launch the facts admit, so the
//!     tiles cover the domain with none left partial.
//!
//! Block `b` owns elements `b * tile .. (b + 1) * tile`. Steps 2 and 3 put
//! every element a thread computes inside the domain, give each element one
//! owner, and leave no element unowned.

use std::collections::HashMap;

use tiki_cute::{Int, Layout, Truth};

use crate::proof::{Proof, ProofError};

/// An elementwise kernel's split of its domain.
///
/// Every tensor of the kernel shares the `domain` layout. Block `b` owns tile
/// `b`, and `threads` maps a (thread, value) coordinate to an element of that
/// tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    /// The layout every tensor of the kernel shares. Its size is the domain.
    pub domain: Layout,
    /// Elements each block owns.
    pub tile: i64,
    /// A rank-2 layout: mode 0 is the thread in the block, mode 1 the value
    /// within the thread.
    pub threads: Layout,
}

impl Partition {
    /// Prove that the grid reaches every element of the domain exactly once.
    pub fn prove(&self) -> Result<Proof, ProofError> {
        let (threads, values) = self.thread_and_value_sizes()?;
        self.check_tile_owners()?;

        // Each tile is owned exactly once; the tiles must now cover the domain
        // with none left partial, for every launch and not only the ones tested.
        let extent = self.domain.size();
        let tile = Int::from(self.tile);
        match extent.is_multiple_of(&tile) {
            Truth::Proven => Ok(Proof::new(extent.div_floor(&tile), threads, values)),
            truth => Err(ProofError::Undivided { tile: self.tile, extent, truth }),
        }
    }

    fn thread_and_value_sizes(&self) -> Result<(i64, i64), ProofError> {
        let sizes: Vec<Option<i64>> =
            self.threads.modes().map(|mode| mode.size().as_static()).collect();
        match sizes[..] {
            [Some(threads), Some(values)] => Ok((threads, values)),
            _ => Err(ProofError::ThreadShape { threads: self.threads.clone() }),
        }
    }

    /// Every tile element has exactly one (thread, value) index.
    fn check_tile_owners(&self) -> Result<(), ProofError> {
        // With as many indices as elements, "no two indices share an element"
        // is the same statement as "every element has one index".
        let size = self.threads.size();
        if size.as_static() != Some(self.tile) {
            return Err(ProofError::TileSize {
                threads: self.threads.clone(),
                size,
                tile: self.tile,
            });
        }

        // A tile is small and static, so checking every index is exact.
        let mut owners: HashMap<i64, i64> = HashMap::new();
        for index in 0..self.tile {
            let offset = self.threads.at(index);
            let inside = |element: &i64| (0..self.tile).contains(element);
            let element = offset.as_int().and_then(Int::as_static).filter(inside);
            let Some(element) = element else {
                return Err(ProofError::OutsideTile {
                    threads: self.threads.clone(),
                    index,
                    offset,
                });
            };
            if let Some(first) = owners.insert(element, index) {
                return Err(ProofError::Overlap {
                    threads: self.threads.clone(),
                    first,
                    second: index,
                    element,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tiki_cute::{Int, Layout, Param, Shape, Truth, Tuple};

    use super::Partition;
    use crate::ProofError;

    /// Elements each block owns in `axpy` from the kernels guide.
    const TILE: i64 = 512;
    /// `axpy`'s split: 128 threads, each owning 4 consecutive elements.
    const THREADS: &str = "(128, 4):(4, 1)";

    fn vector(extent: Int) -> Layout {
        Layout::from(Shape::try_from(Tuple::Leaf(extent)).unwrap())
    }

    fn partition(domain: Layout, threads: &str) -> Partition {
        Partition { domain, tile: TILE, threads: threads.parse().unwrap() }
    }

    // --- Proven partitions ---

    /// Invariant: `axpy`'s partition holds for every `N` the launch facts admit.
    /// Witness: with `N` a multiple of 512 the proof gives `floor(N/512)` blocks
    /// of 128 threads that own 4 values each.
    #[test]
    fn invariant_axpy_partition_is_proven() {
        let n = Int::from(Param::multiple_of("N", TILE).unwrap());
        let proof = partition(vector(n.clone()), THREADS).prove().unwrap();
        assert_eq!(proof.blocks(), &n.div_floor(&Int::from(TILE)));
        assert_eq!((proof.threads(), proof.values()), (128, 4));
    }

    /// Invariant: a proven partition reaches every element of the domain once.
    /// Witness: a 1536-element domain; every (block, thread, value) index,
    /// evaluated through the thread-value layout, lands on a distinct element
    /// and together they reach all 1536.
    #[test]
    fn invariant_every_element_has_one_owner() {
        const EXTENT: i64 = 3 * TILE;
        let partition = partition(vector(Int::from(EXTENT)), THREADS);
        let proof = partition.prove().unwrap();
        let mut owners = vec![0u32; usize::try_from(EXTENT).unwrap()];
        for block in 0..proof.blocks().as_static().unwrap() {
            for index in 0..TILE {
                let within = partition.threads.at(index).as_int().unwrap().as_static().unwrap();
                let element = usize::try_from(block * TILE + within).unwrap();
                owners[element] += 1;
            }
        }
        assert!(owners.iter().all(|&count| count == 1));
    }

    // --- Refused partitions ---

    /// Invariant: the tile must divide the domain for every launch, not some.
    /// Witness: `N` with no divisibility fact is open, and 1000 elements are
    /// refuted outright.
    #[test]
    fn invariant_tile_must_divide_every_launch() {
        let open = partition(vector(Int::from(Param::positive("N"))), THREADS).prove();
        assert!(matches!(open, Err(ProofError::Undivided { truth: Truth::Open, .. })));
        let refuted = partition(vector(Int::from(1000)), THREADS).prove();
        assert!(matches!(refuted, Err(ProofError::Undivided { truth: Truth::Refuted, .. })));
    }

    /// Invariant: no two threads own one tile element.
    /// Witness: offsets `2 * thread + value` send index 1 (thread 1, value 0)
    /// and index 256 (thread 0, value 2) to element 2.
    #[test]
    fn invariant_overlapping_threads_are_refused() {
        let result = partition(vector(Int::from(TILE)), "(128, 4):(2, 1)").prove();
        let expected = (1, 256, 2);
        assert!(matches!(
            result,
            Err(ProofError::Overlap { first, second, element, .. })
                if (first, second, element) == expected
        ));
    }

    /// Invariant: every thread-value index stays inside the tile.
    /// Witness: offsets `8 * thread + value` send index 64 (thread 64) to 512.
    #[test]
    fn invariant_threads_stay_inside_the_tile() {
        let result = partition(vector(Int::from(TILE)), "(128, 4):(8, 1)").prove();
        assert!(matches!(result, Err(ProofError::OutsideTile { index: 64, .. })));
    }

    /// Invariant: the threads cover the whole tile.
    /// Witness: 64 threads of 4 values cover 256 of 512 elements.
    #[test]
    fn invariant_threads_fill_the_tile() {
        let result = partition(vector(Int::from(TILE)), "(64, 4):(4, 1)").prove();
        assert!(matches!(result, Err(ProofError::TileSize { tile: TILE, .. })));
    }

    /// Invariant: a thread-value layout names a thread mode and a value mode.
    /// Witness: the rank-1 layout `512:1` has no value mode.
    #[test]
    fn invariant_threads_need_two_modes() {
        let result = partition(vector(Int::from(TILE)), "512:1").prove();
        assert!(matches!(result, Err(ProofError::ThreadShape { .. })));
    }
}
