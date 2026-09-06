# Backend evaluation reproductions

This directory contains version-pinned procedures for findings F-01 through
F-03 in [ADR-0001](../DECISION-2026-09-05.md). The procedures specify both
successful controls and expected failures, allowing an implementation's
behavior to be checked against the September 5, 2026 evaluation.

Each Rust fixture is an independent Cargo workspace. These procedures exercise
the evaluated dependencies directly and do not require an MLX build.

For the allocator sanitizer checks and the MLX host-export benchmark, see
[CUDA allocator validation](../VALIDATION-2026-09-06.md).

## Preparation

Run the commands from this directory. Use an external directory for compiler
outputs and cloned dependencies:

```sh
repro_output=$(mktemp -d)
export CARGO_TARGET_DIR="$repro_output/target"
```

## cuTile scoped graph lifetime

Requirements: Linux, an NVIDIA GH200 or another compatible GPU, CUDA 13.3, and
Rust 1.92.0 or later. CUDA 13.3 is required for the tested Hopper target. The
manifest pins cuTile to `7f606336e3e8dffa0e1ca3920844082afebb6810`.

```sh
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo run --locked --release --manifest-path cutile_graph_lifetime/Cargo.toml
cargo run --locked --release --manifest-path cutile_graph_lifetime/Cargo.toml -- --drop-input
```

**Expected result.** The first command retains the input and exits 0 after
checking the original prefix sums, whose first and last values are `0` and
`8128`. At the evaluated revision, `--drop-input` forces reuse of the captured
input address and exits
nonzero with this error:

```text
Error: replay changed after dropping input: first=1000, last=128000
```

The mismatch reproduces F-01. If the allocator reports
`allocator did not reuse captured input`, the run is inconclusive because it
has not established the allocation-reuse condition required by the test.

The source and lockfile correspond to the fixture validated on the GH200 for
this record. The program also checks forward and backward scan results and
prints timing information. That output is not a comparative performance
qualification.

The fixture contains no `unsafe` block or raw-memory dereference. It reads a
device address only to verify that the allocator reuses the captured storage.
Run it in an isolated process. A later implementation is acceptable if it
rejects the invalid lifetime or keeps the captured input alive and returns the
correct values.

## CXX ownership and external-type constraint

Requirements: Rust 1.97.1 and a C++20 compiler. The dependencies pin CXX and
`cxx-build` to 1.0.200. The shared fixture is a host-only `Vec<f32>` owner.

```sh
cargo run --locked --manifest-path cxx/Cargo.toml
cargo check --locked --manifest-path cxx_foreign_type/Cargo.toml
```

**Expected result.** The first command exits 0. C++ constructs and mutates an
opaque Rust buffer, then transfers ownership back to Rust for destruction.
The C++ build enables warnings and treats them as errors.

The second command reproduces F-02 and intentionally fails with `E0117`,
because the opaque Rust type is defined in a sibling crate. The control uses
a local newtype. The selected backend architecture places the bridge in the
type-owning crate to avoid that additional adapter.

## Crubit generated-header diagnostics

Requirements: Git and Apple Clang 21.0.0 for the recorded diagnostic. Other
Clang versions can produce different diagnostics. The checked-in header is
the exact generated fixture from the evaluated Crubit revision. It has not
been simplified into handwritten bindings.

```sh
git clone https://github.com/google/crubit.git "$repro_output/crubit"
git -C "$repro_output/crubit" checkout --detach 6e856fe23f0dc6755ee394ead76d4c7894091015
clang++ -std=c++20 -Wall -Wextra -Werror -fsyntax-only \
  -I crubit/generated -I "$repro_output/crubit" crubit/client.cc
```

**Expected result.** The command reproduces F-03 and fails with
`-Wnontrivial-memcall` diagnostics from generated relocation code and Crubit's
support headers. The following control disables only that diagnostic:

```sh
clang++ -std=c++20 -Wall -Wextra -Werror -Wno-nontrivial-memcall \
  -fsyntax-only -I crubit/generated -I "$repro_output/crubit" crubit/client.cc
```

The control succeeds. These syntax checks establish the compiler integration
constraint; they neither link the Rust library nor verify relocation behavior
at runtime.

### Regenerate and run the Crubit fixture

The original generation and linked-client test uses
`nightly-2026-09-04`, its `rustc-dev` component, and the same pinned Crubit
checkout. On macOS:

```sh
rustup toolchain install nightly-2026-09-04 --profile minimal --component rustc-dev
crubit_target="$repro_output/crubit-target"
cargo +nightly-2026-09-04 build --manifest-path "$repro_output/crubit/Cargo.toml" \
  --target-dir "$crubit_target" -p cc_bindings_from_rs -p cargo-cpp_api_from_rust
export PATH="$crubit_target/debug:$PATH"
crubit_sysroot=$(rustup run nightly-2026-09-04 rustc --print sysroot)
export DYLD_LIBRARY_PATH="$crubit_sysroot/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
(
  cd backend_contract
  cargo +nightly-2026-09-04 cpp_api_from_rust --target-dir "$repro_output/generated"
)
clang++ -std=c++20 -Wno-nontrivial-memcall \
  -I "$repro_output/generated/debug/include/crubit" -I "$repro_output/crubit" \
  crubit/client.cc "$repro_output/generated/debug/libtiki-backend-contract.a" \
  -lpthread -ldl -o "$repro_output/crubit-client"
"$repro_output/crubit-client"
```

The linked client exits 0 after checking construction and mutation. The
snapshot-only syntax check requires no Rust nightly installation.

## Recorded verification results

| Fixture | Exit code | Observed result |
| --- | ---: | --- |
| cuTile, retained input | 0 | Original prefix sums match. |
| cuTile, forced reuse after input drop | 1 | `first=1000, last=128000`. |
| CXX ownership control | 0 | C++ construction, mutation, and ownership transfer pass. |
| CXX foreign-type fixture | 101 | Rust reports `E0117`. |
| Crubit pinned header with `-Werror` | 1 | Four `-Wnontrivial-memcall` errors. |
| Crubit pinned header with that warning disabled | 0 | Syntax check passes. |

The expected failures reproduce the identified constraints and defects. Backend
qualification requires the broader checks in the
[architecture specification](../ARCHITECTURE.md#qualification-requirements).
The Crubit generation and linked-client result belongs to the September 5
evaluation; the documentation validation additionally repeats its pinned-header
diagnostic without rebuilding the generator.
