// Copyright (c) 2026 Dedalus Labs, Inc. All rights reserved.

//! Build the C++ client against the generated CXX bridge.

fn main() {
    cxx_build::bridge("src/main.rs")
        .file("client.cc")
        .include(".")
        .std("c++20")
        .warnings(true)
        .warnings_into_errors(true)
        .compile("tiki-cxx-client");
    for path in ["src/main.rs", "client.cc", "client.h"] {
        println!("cargo:rerun-if-changed={path}");
    }
}
