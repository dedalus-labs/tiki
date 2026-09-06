// Copyright (c) 2026 Dedalus Labs, Inc. All rights reserved.

//! This fixture intentionally fails: the opaque type belongs to another crate.

use tiki_backend_contract::Buffer;

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        type Buffer;
    }
}
