// Copyright © 2026 Dedalus Labs, Inc.

//! Link the extension module so that the interpreter that loads it supplies the Python symbols.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
