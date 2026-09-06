# Cooperative scheduling proof

Validated September 4, 2026 on NVIDIA GH200 144 GB HBM3e, CUDA 13.3,
CuTe DSL 4.7.1, and the existing MLX CUDA build from upstream commit
`b6368984b8e02a3fb3ee7986846c0fb85e1fccf7`. No MLX C++ changes were made.

## Results

All 18 tests in `test_compile.py` and `test_cooperative.py` pass on the GH200.
The Mac passes 12 capture/layout tests and skips six CUDA execution tests.
RMSNorm covers row widths 1, 31, 129, 1024, and 4096 with four thread schedules.
Transpose covers full, partial, and empty tiles with three swizzle settings.

The deterministic demo uses 5x129 inputs for RMSNorm and a 33x65 matrix for
transpose:

| Schedule | Threads/block | Shared scratch | Maximum error | Cubin size |
| --- | ---: | ---: | ---: | ---: |
| RMSNorm: 32 threads/row, 4 rows/block | 128 | 0 bytes | 4.7684e-7 | 4032 bytes |
| RMSNorm: 64 threads/row, 2 rows/block | 128 | 16 bytes | 4.7684e-7 | 4800 bytes |
| Transpose: plain layout | 128 | 4096 bytes | 0 | 4792 bytes |
| Transpose: `S<5,0,5>` | 128 | 4096 bytes | 0 | 4792 bytes |

The emitted PTX contains warp shuffle instructions for the reductions, plus
shared-memory stores, loads, and a block barrier for multi-warp rows.
The swizzled transpose contains explicit XOR address calculations surrounding
shared-memory loads and stores. The layout choice survives lowering.

This demonstrates scheduling control and correctness. Kernel performance and
compiler-overhead comparisons have not been measured in this experiment.

## Issues exposed

**Primitive parameters matter.** MLX exports sums as `Reduce` with a reduction
kind and axes, and reciprocal square root as `Sqrt` with a flag. The graph
reader now checks these parameters and records the supported semantics as
`ReduceSum` and `Rsqrt`. Other axes/kinds are rejected.

**A one-element row loses its reduction node.** MLX simplifies that sum away.
The initial implementation rejected width-one RMSNorm. A focused test reproduced
the failure; the row schedule now accepts that simplified graph and emits its
arithmetic without shuffles or shared scratch.

**Partial rows must still reach the block barrier.** Inactive rows contribute
zero partials and skip output stores. Their threads participate in the same
barrier as the active rows. No early return can strand other warps.

## Device diagnostics and their limit

Compute Sanitizer memcheck and racecheck both pass with instrumentation
restricted to the generated kernel, `tiki_fused`:

```sh
compute-sanitizer --tool memcheck --kernel-name kne=tiki_fused --error-exitcode 1 \
  python -m unittest discover -s experiments/cute_backend -p test_cooperative.py
compute-sanitizer --tool racecheck --kernel-name kne=tiki_fused --error-exitcode 1 \
  python -m unittest discover -s experiments/cute_backend -p test_cooperative.py
```

Memcheck reports zero errors. Racecheck reports zero errors and zero warnings.

The **unfiltered** run fails with a potential use-before-allocation report in
MLX kernels. The failure also reproduces with this independent program, which
does not import Tiki:

```python
import mlx.core as mx
mx.eval(mx.random.normal((5, 1)))
```

Under unfiltered memcheck, the minimal program reports a read that may precede
a stream-ordered allocation, then a CUDA launch failure. This remains an
unresolved MLX/runtime/tooling issue. The filtered passes do not establish that
the whole MLX CUDA runtime is sanitizer-clean; no checks were disabled in the
unfiltered repro.

## Layout inspection

The optional inspector loads the original monorepo `debug.py` from
`packages/python/tiki/src/tiki/kernels/cute/lib/` at commit
`7858ecd1aea016156a5df3eef36d40fbe5791892`.
Its Git blob is `94286193112eaf051cc63e385787b3151393b1fb`.
The visualizer and validator helpers were merged in
[PR #2586](https://github.com/dedalus-labs/dedalus/pull/2586), June 20, 2026 Pacific
time (June 21 UTC).

Before drawing, the inspector evaluates actual CuTe thread/value layouts and
checks all 1024 logical tile coordinates against the emitted load and read
assignments. The visualizer uses column-major flattened positions, so the
thread/value strides account for that convention explicitly.

For the scalar 32x32 float32 transpose, a warp's column read addresses one bank
in the plain layout and all 32 banks with `S<5,0,5>`. The bank diagram shows this
address permutation; it does not measure conflict counters or claim a speedup.
