// Copyright © 2026 Dedalus Labs, Inc.

//! Generate the optional CXX boundary without adding a CUDA dependency.

fn main() {
    #[cfg(feature = "cxx-bridge")]
    {
        cxx_build::bridge("src/bridge.rs").std("c++17").compile("tiki-layout-bridge");
        println!("cargo:rerun-if-changed=src/bridge.rs");
    }
}
