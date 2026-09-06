// Copyright © 2026 Dedalus Labs, Inc.

//! Size-class cache of released storage with oldest-first eviction.
//!
//! Reuse takes the smallest class that fits when it is below
//! `min(2 * size, size + 2 * page)`; eviction releases the least recently
//! recycled entry across all classes. These are MLX's BufferCache rules.

use std::collections::{BTreeMap, VecDeque};

pub struct SizeClassCache<A> {
    page: usize,
    classes: BTreeMap<usize, VecDeque<(u64, A)>>,
    order: BTreeMap<u64, usize>,
    seq: u64,
    bytes: usize,
}

impl<A> SizeClassCache<A> {
    pub fn new(page: usize) -> Self {
        Self {
            page,
            classes: BTreeMap::new(),
            order: BTreeMap::new(),
            seq: 0,
            bytes: 0,
        }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn reuse(&mut self, size: usize) -> Option<A> {
        let limit = (2 * size).min(size + 2 * self.page);
        let class = *self.classes.range(size..).next()?.0;
        if class >= limit {
            return None;
        }
        let (seq, a) = self.pop_front(class);
        self.order.remove(&seq);
        self.bytes -= class;
        Some(a)
    }

    pub fn recycle(&mut self, size: usize, a: A) {
        self.seq += 1;
        self.classes
            .entry(size)
            .or_default()
            .push_back((self.seq, a));
        self.order.insert(self.seq, size);
        self.bytes += size;
    }

    /// Release at least `min_bytes`, or everything when that is most of the cache.
    pub fn release(&mut self, min_bytes: usize, free: &mut dyn FnMut(A)) -> usize {
        if min_bytes as f64 >= 0.9 * self.bytes as f64 {
            return self.clear(free);
        }
        let mut released = 0;
        let mut count = 0;
        while released < min_bytes {
            let Some((seq, class)) = self.order.pop_first() else {
                break;
            };
            let (front, a) = self.pop_front(class);
            debug_assert_eq!(front, seq);
            released += class;
            count += 1;
            free(a);
        }
        self.bytes -= released;
        count
    }

    pub fn clear(&mut self, free: &mut dyn FnMut(A)) -> usize {
        let mut count = 0;
        for (_, queue) in std::mem::take(&mut self.classes) {
            for (_, a) in queue {
                count += 1;
                free(a);
            }
        }
        self.order.clear();
        self.bytes = 0;
        count
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
        assert_eq!(cache.reuse(60), Some("a"));
        assert_eq!(cache.reuse(60), Some("b"));
        assert_eq!(cache.reuse(60), Some("c"));
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
        assert_eq!(cache.reuse(100), Some("y"));
    }

    // Invariant: release evicts the least recently recycled entries across classes.
    // Witness: recycle 64 then 128 then 64; releasing 64 bytes frees only the first.
    #[test]
    fn release_evicts_oldest_across_classes() {
        let mut cache = SizeClassCache::new(PAGE);
        cache.recycle(64, "old");
        cache.recycle(128, "mid");
        cache.recycle(64, "new");
        let mut freed = Vec::new();
        assert_eq!(cache.release(64, &mut |a| freed.push(a)), 1);
        assert_eq!(freed, ["old"]);
        assert_eq!(cache.bytes(), 192);
        assert_eq!(cache.reuse(64), Some("new"));
    }

    // Invariant: a request for at least 90% of the cache clears it entirely.
    // Witness: 200 cached bytes; releasing 180 frees both entries.
    #[test]
    fn release_of_most_bytes_clears() {
        let mut cache = SizeClassCache::new(PAGE);
        cache.recycle(100, 1);
        cache.recycle(100, 2);
        let mut freed = Vec::new();
        assert_eq!(cache.release(180, &mut |a| freed.push(a)), 2);
        assert_eq!(cache.bytes(), 0);
        assert!(cache.reuse(100).is_none());
    }
}
