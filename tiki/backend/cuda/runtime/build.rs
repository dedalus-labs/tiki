// Copyright © 2026 Dedalus Labs, Inc.

fn main() {
    let mut build = cxx_build::bridge("src/bridge.rs");
    if let Ok(paths) = std::env::var("CUDA_INCLUDE_DIRS") {
        for path in paths.split(';') {
            build.include(path);
        }
    } else if let Ok(path) = std::env::var("CUDA_TOOLKIT_PATH") {
        build.include(format!("{path}/include"));
    }
    build
        .file("src/cudart.cpp")
        .std("c++20")
        .warnings(true)
        .warnings_into_errors(true)
        .compile("tiki_cuda_runtime_bridge");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/cudart.cpp");
    println!("cargo:rerun-if-env-changed=CUDA_INCLUDE_DIRS");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    if let Ok(path) = std::env::var("CUDA_TOOLKIT_PATH") {
        println!("cargo:rustc-link-search=native={path}/lib64");
    }
    println!("cargo:rustc-link-lib=cudart");
}
