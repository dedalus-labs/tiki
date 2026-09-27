// Copyright © 2026 Dedalus Labs, Inc.

//! Size-class cache of released storage with oldest-first eviction.
//!
//! Reuse takes the smallest class that fits when it is below
//! `min(2 * size, size + 2 * page)`; eviction releases the least recently
//! recycled entry across all classes. These are the rules of Tiki's C++ `BufferCache`.

use std::collections::{BTreeMap, VecDeque};

pub struct SizeClassCache<A> {
    page: usize,
    classes: BTreeMap<usize, VecDeque<(u64, A)>>,
    seq: u64,
    bytes: usize,
}

impl<A> SizeClassCache<A> {
    pub fn new(page: usize) -> Self {
        Self { page, classes: BTreeMap::new(), seq: 0, bytes: 0 }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn reuse(&mut self, size: usize) -> Option<(usize, A)> {
        let limit = (2 * size).min(size + 2 * self.page);
        let class = *self.classes.range(size..).next()?.0;
        if class >= limit {
            return None;
        }
        let (_, a) = self.pop_front(class);
        self.bytes -= class;
        Some((class, a))
    }

    pub fn recycle(&mut self, size: usize, a: A) {
        self.seq += 1;
        self.classes.entry(size).or_default().push_back((self.seq, a));
        self.bytes += size;
    }

    /// Remove at least `min_bytes`, or everything when that is most of the
    /// cache, and return the removed entries oldest first.
    pub fn release(&mut self, min_bytes: usize) -> Vec<A> {
        if 10 * min_bytes >= 9 * self.bytes {
            return self.clear();
        }
        let mut released = 0;
        let mut removed = Vec::new();
        while released < min_bytes {
            let Some(class) = self.oldest_class() else {
                break;
            };
            let (_, a) = self.pop_front(class);
            released += class;
            removed.push(a);
        }
        self.bytes -= released;
        removed
    }

    /// Remove every entry and return them.
    pub fn clear(&mut self) -> Vec<A> {
        self.bytes = 0;
        std::mem::take(&mut self.classes)
            .into_values()
            .flat_map(|queue| queue.into_iter().map(|(_, a)| a))
            .collect()
    }

    /// Class whose oldest entry is the oldest overall. Eviction only runs under
    /// memory pressure, so a scan beats a second index maintained on every reuse.
    fn oldest_class(&self) -> Option<usize> {
        self.classes
            .iter()
            .min_by_key(|(_, queue)| queue.front().map(|entry| entry.0))
            .map(|(class, _)| *class)
    }

    fn pop_front(&mut self, class: usize) -> (u64, A) {
        let queue = self.classes.get_mut(&class).expect("class is indexed");
        let entry = queue.pop_front().expect("class is non-empty");
        if queue.is_empty() {
            self.classes.remove(&class);
        }
        entry
    }
}

#[cfg(test)]
mod tests {
    use super::SizeClassCache;

    const PAGE: usize = 16;

    // Invariant: reuse returns the smallest fitting class, oldest entry first.
    // Witness: two 64-byte entries and one 80-byte entry; a 60-byte request
    // takes the first 64-byte entry, then the second, then the 80-byte one.
    #[test]
    fn reuse_prefers_smallest_class_then_oldest() {
        let mut cache = SizeClassCache::new(PAGE);
        cache.recycle(80, "c");
        cache.recycle(64, "a");
        cache.recycle(64, "b");
        assert_eq!(cache.reuse(60), Some((64, "a")));
        assert_eq!(cache.reuse(60), Some((64, "b")));
        assert_eq!(cache.reuse(60), Some((80, "c")));
        assert_eq!(cache.reuse(60), None);
        assert_eq!(cache.bytes(), 0);
    }

    // Invariant: a class at or beyond min(2 * size, size + 2 * page) is not reused.
    // Witness: page 16, request 100 gives limit 132; a 132-byte entry stays cached.
    #[test]
    fn reuse_rejects_oversized_class() {
        let mut cache = SizeClassCache::new(PAGE);
        cache.recycle(132, "x");
        assert_eq!(cache.reuse(100), None);
        cache.recycle(131, "y");
        assert_eq!(cache.reuse(100), Some((131, "y")));
    }

    // Invariant: release evicts the least recently recycled entries across classes.
    // Witness: recycle 64 then 128 then 64; releasing 64 bytes frees only the first.
    #[test]
    fn release_evicts_oldest_across_classes() {
        let mut cache = SizeClassCache::new(PAGE);
        cache.recycle(64, "old");
        cache.recycle(128, "mid");
        cache.recycle(64, "new");
        assert_eq!(cache.release(64), ["old"]);
        assert_eq!(cache.bytes(), 192);
        assert_eq!(cache.reuse(64), Some((64, "new")));
    }

    // Invariant: a request for at least 90% of the cache clears it entirely.
    // Witness: 200 cached bytes; releasing 180 frees both entries.
    #[test]
    fn release_of_most_bytes_clears() {
        let mut cache = SizeClassCache::new(PAGE);
        cache.recycle(100, 1);
        cache.recycle(100, 2);
        assert_eq!(cache.release(180).len(), 2);
        assert_eq!(cache.bytes(), 0);
        assert!(cache.reuse(100).is_none());
    }
}
