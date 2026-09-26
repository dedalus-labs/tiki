# Rust CUDA backend

Tiki's execution is memory safe from its Python API to the GPU. Rust owns all
host code and every GPU memory address, and C++ remains only as arithmetic
inside GPU kernels that calls Rust accessors for each load and store.
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

## Implementation status

The port is tracked in [#65](https://github.com/dedalus-labs/tiki/issues/65).
Execution uses the MLX CUDA runtime until each component passes the
qualification gates in ADR-0001.
