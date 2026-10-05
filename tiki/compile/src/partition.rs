// Copyright © 2026 Dedalus Labs, Inc.

//! How an elementwise kernel splits its domain across blocks and threads.
//!
//! [`Partition::new`] proves, before a partition exists:
//!
//!  1. The thread-value layout has a thread mode and a value mode, each of
//!     static size. Their product is the tile, the elements one block owns.
//!  2. It reaches each element of the tile exactly once: every (thread, value)
//!     index lands on a distinct element inside the tile.
//!  3. The tile divides the domain for every launch the facts admit, so the
//!     tiles cover the domain with none left partial.
//!
//! Block `b` owns elements `b * tile .. (b + 1) * tile`. Steps 2 and 3 put
//! every element a thread computes inside the domain, give each element one
//! owner, and leave no element unowned.

use std::collections::HashMap;

use tiki_cute::{Int, Layout, Truth};

use crate::PartitionError;

/// An elementwise kernel's split of its domain across blocks and threads.
///
/// Every tensor of the kernel shares the domain layout. Block `b` owns tile
/// `b`, and the thread-value layout maps a (thread, value) coordinate to an
/// element of that tile. Only [`Partition::new`] makes one, so holding a
/// `Partition` means its grid reaches every element of the domain exactly
/// once, for every launch the domain's facts admit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    domain: Layout,
    thread_values: Layout,
    /// The domain's size over the tile, exact for every admitted launch
    /// because the tile divides the domain.
    blocks: Int,
    threads: i64,
    values: i64,
}

impl Partition {
    /// Split `domain` into tiles that `thread_values` spreads across one
    /// block's threads, or name the step of the proof that fails.
    ///
    /// `thread_values` is a rank-2 layout: mode 0 is the thread in the block,
    /// mode 1 the value within the thread.
    pub fn new(domain: Layout, thread_values: Layout) -> Result<Partition, PartitionError> {
        let (threads, values) = thread_and_value_sizes(&thread_values)?;
        let tile = threads * values;
        check_tile_owners(&thread_values, tile)?;

        // Each tile element has one owner; the tiles must now cover the domain
        // with none left partial, for every launch and not only the ones tested.
        let extent = domain.size();
        let divisor = Int::from(tile);
        match extent.is_multiple_of(&divisor) {
            Truth::Proven => {
                let blocks = extent.div_floor(&divisor);
                Ok(Partition { domain, thread_values, blocks, threads, values })
            }
            truth => Err(PartitionError::Undivided { tile, extent, truth }),
        }
    }

    /// The layout every tensor of the kernel shares. Its size is the domain.
    pub fn domain(&self) -> &Layout {
        &self.domain
    }

    /// The layout from a (thread, value) coordinate to an element of the
    /// block's tile.
    pub fn thread_values(&self) -> &Layout {
        &self.thread_values
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

fn thread_and_value_sizes(thread_values: &Layout) -> Result<(i64, i64), PartitionError> {
    let sizes: Vec<Option<i64>> =
        thread_values.modes().map(|mode| mode.size().as_static()).collect();
    match sizes[..] {
        [Some(threads), Some(values)] => Ok((threads, values)),
        _ => Err(PartitionError::ThreadShape { thread_values: thread_values.clone() }),
    }
}

/// Every tile element has exactly one (thread, value) index.
fn check_tile_owners(thread_values: &Layout, tile: i64) -> Result<(), PartitionError> {
    // The layout has as many indices as the tile has elements, so "no two
    // indices share an element" is the same statement as "every element has
    // one index". A tile is small and static, so checking every index is exact.
    let mut owners: HashMap<i64, i64> = HashMap::new();
    for index in 0..tile {
        let offset = thread_values.at(index);
        let inside = |element: &i64| (0..tile).contains(element);
        let element = offset.as_int().and_then(Int::as_static).filter(inside);
        let Some(element) = element else {
            return Err(PartitionError::OutsideTile {
                thread_values: thread_values.clone(),
                index,
                offset,
            });
        };
        if let Some(first) = owners.insert(element, index) {
            return Err(PartitionError::Overlap {
                thread_values: thread_values.clone(),
                first,
                second: index,
                element,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tiki_cute::{Int, Layout, Param, Shape, Truth, Tuple};

    use super::Partition;
    use crate::PartitionError;

    /// Elements each block owns in `axpy` from the kernels guide.
    const TILE: i64 = 512;
    /// `axpy`'s split: 128 threads, each owning 4 consecutive elements.
    const THREAD_VALUES: &str = "(128, 4):(4, 1)";

    fn vector(extent: Int) -> Layout {
        Layout::from(Shape::try_from(Tuple::Leaf(extent)).unwrap())
    }

    fn split(domain: Layout, thread_values: &str) -> Result<Partition, PartitionError> {
        Partition::new(domain, thread_values.parse().unwrap())
    }

    // --- Proven partitions ---

    /// Invariant: `axpy`'s partition holds for every `N` the launch facts admit.
    /// Witness: with `N` a multiple of 512 the proof gives `floor(N/512)` blocks
    /// of 128 threads that own 4 values each.
    #[test]
    fn invariant_axpy_partition_is_proven() {
        let n = Int::from(Param::multiple_of("N", TILE).unwrap());
        let partition = split(vector(n.clone()), THREAD_VALUES).unwrap();
        assert_eq!(partition.blocks(), &n.div_floor(&Int::from(TILE)));
        assert_eq!((partition.threads(), partition.values()), (128, 4));
    }

    /// Invariant: a proven partition reaches every element of the domain once.
    /// Witness: a 1536-element domain; every (block, thread, value) index,
    /// evaluated through the thread-value layout, lands on a distinct element
    /// and together they reach all 1536.
    #[test]
    fn invariant_every_element_has_one_owner() {
        const EXTENT: i64 = 3 * TILE;
        let partition = split(vector(Int::from(EXTENT)), THREAD_VALUES).unwrap();
        let mut owners = vec![0u32; usize::try_from(EXTENT).unwrap()];
        for block in 0..partition.blocks().as_static().unwrap() {
            for index in 0..TILE {
                let offset = partition.thread_values().at(index);
                let within = offset.as_int().unwrap().as_static().unwrap();
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
        let open = split(vector(Int::from(Param::positive("N"))), THREAD_VALUES);
        assert!(matches!(open, Err(PartitionError::Undivided { truth: Truth::Open, .. })));
        let refuted = split(vector(Int::from(1000)), THREAD_VALUES);
        assert!(matches!(refuted, Err(PartitionError::Undivided { truth: Truth::Refuted, .. })));
    }

    /// Invariant: no two threads own one tile element.
    /// Witness: offsets `2 * thread + value` send index 1 (thread 1, value 0)
    /// and index 256 (thread 0, value 2) to element 2.
    #[test]
    fn invariant_overlapping_threads_are_refused() {
        let result = split(vector(Int::from(TILE)), "(128, 4):(2, 1)");
        let expected = (1, 256, 2);
        assert!(matches!(
            result,
            Err(PartitionError::Overlap { first, second, element, .. })
                if (first, second, element) == expected
        ));
    }

    /// Invariant: every thread-value index stays inside the tile.
    /// Witness: offsets `8 * thread + value` send index 64 (thread 64) to 512.
    #[test]
    fn invariant_threads_stay_inside_the_tile() {
        let result = split(vector(Int::from(TILE)), "(128, 4):(8, 1)");
        assert!(matches!(result, Err(PartitionError::OutsideTile { index: 64, .. })));
    }

    /// Invariant: a thread-value layout names a thread mode and a value mode.
    /// Witness: the rank-1 layout `512:1` has no value mode.
    #[test]
    fn invariant_threads_need_two_modes() {
        let result = split(vector(Int::from(TILE)), "512:1");
        assert!(matches!(result, Err(PartitionError::ThreadShape { .. })));
    }
}
