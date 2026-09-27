// Copyright © 2026 Dedalus Labs, Inc.

//! Block bookkeeping for the fixed-size small pool.

pub struct FreeList {
    free: Vec<u32>,
}

impl FreeList {
    pub fn new(blocks: u32) -> Self {
        Self {
            free: (0..blocks).rev().collect(),
        }
    }

    pub fn take(&mut self) -> Option<u32> {
        self.free.pop()
    }

    pub fn give(&mut self, block: u32) {
        self.free.push(block);
    }
}

#[cfg(test)]
mod tests {
    use super::FreeList;

    // Invariant: blocks are handed out in ascending order and reused last-in first-out.
    // Witness: three blocks; giving back 1 makes 1 the next block, then the pool runs dry.
    #[test]
    fn ascending_then_lifo() {
        let mut list = FreeList::new(3);
        assert_eq!(list.take(), Some(0));
        assert_eq!(list.take(), Some(1));
        list.give(1);
        assert_eq!(list.take(), Some(1));
        assert_eq!(list.take(), Some(2));
        assert_eq!(list.take(), None);
    }
}
