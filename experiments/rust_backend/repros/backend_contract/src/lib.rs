// Copyright (c) 2026 Dedalus Labs, Inc. All rights reserved.

//! Host-only ownership fixture for comparing generated C++ interfaces.

pub struct Buffer {
    values: Vec<f32>,
}

impl Buffer {
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            values: vec![0.0; size],
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn fill(&mut self, value: f32) {
        self.values.fill(value);
    }

    #[must_use]
    pub fn checksum(&self) -> f32 {
        self.values.iter().sum()
    }
}

#[must_use]
pub fn consume(buffer: Buffer) -> usize {
    buffer.len()
}
