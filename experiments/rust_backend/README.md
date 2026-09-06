# Rust CUDA backend

Tiki's CUDA backend architecture assigns resource ownership and asynchronous
execution to Rust, kernel compilation to the CuTe toolchain, and model
semantics and automatic differentiation to the MLX graph layer. A CXX bridge
connects the existing C++ core to an explicit Rust runtime interface.

Rust is selected to make allocation ownership, permitted access, and resource
retirement enforceable through a controlled API. The runtime must preserve
these contracts until device work completes, including across graph replay,
cancellation, and cache eviction. The compiler and runtime communicate through
kernel artifacts and argument contracts, allowing each to evolve independently.

## Documentation

| Document | Purpose |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | Responsibilities, Rust design principles, ownership contracts, and compatibility requirements. |
| [Architecture decision ADR-0001](DECISION-2026-09-05.md) | Rationale, alternatives, and consequences of the decision accepted on September 5, 2026. |
| [Evaluation reproductions](repros/README.md) | Version-pinned procedures and observed results supporting the decision. |
| [Allocator validation](VALIDATION-2026-09-06.md) | Scoped GPU checks, test counts, and cache-controlled export measurements. |

## Implementation status

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

Kernel execution still uses the MLX CUDA command encoder through the
[existing CuTe compiler path](../cute_backend/README.md). Submission
retention, completion tracking, and graph replay ownership are the next
migration steps in ADR-0001. Adoption of cuTile runtime components requires
the ownership qualification specified there.
