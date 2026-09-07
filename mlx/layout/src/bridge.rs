// Copyright © 2026 Dedalus Labs, Inc.

//! CXX transports owned, validated swizzles without exposing storage pointers.

use crate::{LayoutError, Swizzle};

#[allow(unsafe_code)] // Only CXX-generated ABI functions use unsafe operations.
#[cxx::bridge(namespace = "mlx::core::layout_rt")]
mod ffi {
    extern "Rust" {
        type Swizzle;
        fn new_swizzle(bits: i64, base: i64, shift: i64) -> Result<Box<Swizzle>>;
        fn bits(self: &Swizzle) -> u32;
        fn base(self: &Swizzle) -> u32;
        fn shift(self: &Swizzle) -> i32;
        fn apply(self: &Swizzle, index: i64) -> Result<i64>;
    }
}

fn new_swizzle(bits: i64, base: i64, shift: i64) -> Result<Box<Swizzle>, LayoutError> {
    Swizzle::new(bits, base, shift).map(Box::new)
}
