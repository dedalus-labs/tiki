// Copyright © 2026 Dedalus Labs, Inc.

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    // The toolkit's stub resolves libcuda at link time; the installed driver provides it at run time.
    if let Ok(path) = std::env::var("CUDA_TOOLKIT_PATH") {
        println!("cargo:rustc-link-search=native={path}/lib64/stubs");
    }
    println!("cargo:rustc-link-lib=cuda");
}
