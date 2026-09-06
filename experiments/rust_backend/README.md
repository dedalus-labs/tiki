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

## Implementation status

The architecture is accepted; integration of the Rust execution runtime is
pending. Current execution uses the MLX CUDA runtime through the
[existing CuTe compiler path](../cute_backend/README.md). Adoption of cuTile
runtime components requires the ownership qualification specified in ADR-0001.
