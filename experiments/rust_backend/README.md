# Rust CUDA backend

Tiki's execution is memory safe from its Python API to the GPU. Rust owns all
host code, and GPU kernels are written in Tiki's kernel language, which follows
CuTe's semantics and has every memory access proven by the compiler. The Rust
stack is built beside the C++ backend and replaces it in one cutover.
[ADR-0001](DECISION-2026-09-05.md) records the decision and its evidence.

Rust makes allocation ownership, permitted access, and resource retirement part
of one checked API. The runtime keeps each resource alive until the device work
that uses it completes, including across graph replay, cancellation, and cache
eviction.

## Documentation

| Document | Purpose |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | Responsibilities, Rust design principles, ownership contracts, and compatibility requirements. |
| [Architecture decision ADR-0001](DECISION-2026-09-05.md) | Decision, kernel placement, enforcement, and evidence. |
| [Evaluation reproductions](repros/README.md) | Version-pinned procedures and observed results supporting the decision. |
| [Allocator validation](VALIDATION-2026-09-06.md) | Scoped GPU checks, test counts, and cache-controlled export measurements. |

## Implementation status

The replacement is tracked in [#65](https://github.com/dedalus-labs/tiki/issues/65).
[`tiki/runtime/cuda`](../../tiki/runtime/cuda) is the Rust CUDA runtime:
size classes, the cache of released memory, memory limits, and batch
completion. It forbids `unsafe`;
[`tiki/runtime/cuda-sys`](../../tiki/runtime/cuda-sys) is its only call site
into `libcuda` and hands out owned streams, events, and device memory. The C++ backend keeps its own allocator until the
cutover.
