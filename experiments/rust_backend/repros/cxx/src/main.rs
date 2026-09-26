// Copyright (c) 2026 Dedalus Labs, Inc. All rights reserved.

//! C++ exercises Rust-owned buffers through an explicit CXX contract.

struct Buffer(tiki_backend_contract::Buffer);

impl Buffer {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn fill(&mut self, value: f32) {
        self.0.fill(value);
    }

    fn checksum(&self) -> f32 {
        self.0.checksum()
    }
}

#[cxx::bridge(namespace = "tiki")]
mod ffi {
    extern "Rust" {
        type Buffer;
        fn new_buffer(size: usize) -> Box<Buffer>;
        fn len(self: &Buffer) -> usize;
        fn fill(self: &mut Buffer, value: f32);
        fn checksum(self: &Buffer) -> f32;
        fn consume_buffer(buffer: Box<Buffer>) -> usize;
    }
    // SAFETY: the client uses the generated buffer API and returns a scalar.
    unsafe extern "C++" {
        include!("client.h");
        fn run_client() -> usize;
    }
}

fn new_buffer(size: usize) -> Box<Buffer> {
    Box::new(Buffer(tiki_backend_contract::Buffer::new(size)))
}

fn consume_buffer(buffer: Box<Buffer>) -> usize {
    tiki_backend_contract::consume(buffer.0)
}

fn main() {
    assert_eq!(ffi::run_client(), 128);
    println!("CXX: construction, mutation, ownership transfer, destruction: PASS");
}
