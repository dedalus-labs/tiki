# Rust CUDA backend

Tiki's execution is memory safe from its Python API to the GPU. Rust owns all
host code, and GPU kernels are written in Tiki's kernel language, which follows
CuTe's semantics and has every memory access proven by the compiler. Inherited
C++ kernels call Rust accessors for each load and store until they are ported.
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

The port is tracked in [#65](https://github.com/dedalus-labs/tiki/issues/65).
The Rust runtime owns CUDA storage: the crate in
[`mlx/backend/cuda/runtime`](../../mlx/backend/cuda/runtime) implements
allocation, size classes, the small pool, the cache, memory limits, and
migration of device storage to unified memory. Migration enqueues the copy and
the release of the device source on one stream, so the source outlives the
copy by construction. Building the CUDA backend requires `cargo` 1.92 or later
on the path; CMake invokes it and links the resulting static library.

This guarantee covers the migration copy and its source release. It does not
establish completion of producers on other streams or extend buffer ownership
through arbitrary asynchronous work. Callers retain those responsibilities.

Kernel execution still uses the MLX CUDA command encoder. Submission
retention, completion tracking, and graph replay ownership move to Rust next,
each once it passes the qualification gates in ADR-0001.
