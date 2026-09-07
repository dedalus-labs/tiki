# Rust indexing boundary

`tiki-layout` owns checked, storage-independent indexing values. Its first
slice is an immutable `Swizzle`: constructor validation and offset evaluation
live in safe Rust, and a narrow CXX bridge exposes the same value to Python.
It does not own arrays, allocations, CUDA streams, or kernel execution.

This is the first migration boundary, not the intended final layout engine.
Hierarchical layout algebra currently remains in the pinned PyCuTe reference.
The next slice moves hierarchical shape and stride trees, their congruence
checks, and checked coordinate-to-offset evaluation into this crate. Composition
expressions and addressed-interval validation follow that common representation.
Python will delegate those operations instead of retaining a second authoritative
implementation.

Each migration must preserve the reference's scalar, flat, and nested coordinate
semantics, including zero and negative strides. Overflow and out-of-bounds
addresses must fail before constructing a view. Composition must preserve the
placement of offsets inside a swizzle. These are acceptance conditions for later
slices, not claims about functionality in this crate today. General GPU lowering
and storage ownership remain separate work.

## Build contract

Tiki requires Rust 1.92 or newer and Cargo for Python source builds on every
backend, including CPU and Metal. The indexing extension is backend-independent,
so its semantics and validation do not change with the device. There is no Python
substitute when the Rust extension is missing. Prebuilt wheel consumers do not
need the Rust toolchain.

C++ CPU and Metal builds with `MLX_BUILD_PYTHON_BINDINGS=OFF` do not build this
extension. The CUDA runtime has its own Rust dependency. `lib.rs` only declares
modules and exports. Implementation belongs in named modules, and CXX transports
validated values across the existing C++ boundary.
