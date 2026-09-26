# CuTe exemplar audit

The supported MLX tensor programs compile to native CUDA device code. On the
GH200, eight of ten measured RMSNorm shape/schedule combinations were within
3% of the same scalar algorithm built with NVCC, NVIDIA's CUDA C++ compiler. Two narrow-row
schedules were 11% and 22% slower. Native compilation is established; optimal
kernel generation and full-framework performance are not.

## References examined

The audit started with the monorepo's CuTe `RowSum`, layout, profiler, and
performance-accounting code. External repositories were cloned into `/tmp`:

- NVIDIA CUTLASS: `59e3a3338d516ca6ce0e073af8da65289678a35c`.
  Read the RMSNorm and elementwise examples, the benchmarking helper, and the
  [staged code-generation model](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html).
- QuACK: `14b0b7122e252a4df40a0b762c7ffd0f8b67a015`.
  Read RMSNorm, reduction configuration, copy setup, cache fingerprints, and tests.

These source snapshots were reviewed, not benchmarked as complete libraries.
The measured backend remained the installed `nvidia-cutlass-dsl==4.7.1`.

## Findings

| Gap | Exemplar | Our status | Fix or next experiment |
| --- | --- | --- | --- |
| Narrow rows waste lanes with a whole warp per row. | [NVIDIA RMSNorm configuration, lines 221–234](https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL/cute/blackwell/kernel/rmsnorm/rmsnorm.py#L221), [QuACK configuration, lines 52–56](https://github.com/Dao-AILab/quack/blob/14b0b7122e252a4df40a0b762c7ffd0f8b67a015/quack/rmsnorm_config.py#L52) | Fixed for explicit schedules. | Added 8- and 16-thread row groups. The 8192x16 case improved by about 26% with 16 threads per row. |
| Invalid operations can escape validation after a reduction simplifies away. | Explicit eligibility checks in [QuACK's reduction setup, lines 42–49](https://github.com/Dao-AILab/quack/blob/14b0b7122e252a4df40a0b762c7ffd0f8b67a015/quack/reduction_base.py#L42); our audit found the specific counterexample. | Fixed. | A single-column `(x * x).T` under `RowSchedule` raised `KeyError`. A fail-first test now requires `UnsupportedGraphError`; operation validation precedes the degenerate-row case. |
| Our memory accesses remain scalar and conservatively aligned. | [NVIDIA's 128-bit copy atoms, lines 434–449](https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL/cute/blackwell/kernel/rmsnorm/rmsnorm.py#L434) | Open optimization. | Add vectorized copies with proven pointer alignment and valid tails. Contiguous offset views can still be misaligned; added numerical coverage for offset and padded row views before changing loads. |
| We reread inputs after the reduction. | [QuACK's register fragments, lines 250–255](https://github.com/Dao-AILab/quack/blob/14b0b7122e252a4df40a0b762c7ffd0f8b67a015/quack/rmsnorm.py#L250), [register/shared reload policy, lines 84–90](https://github.com/Dao-AILab/quack/blob/14b0b7122e252a4df40a0b762c7ffd0f8b67a015/quack/rmsnorm_config.py#L84) | Open optimization. | Compare retaining values in registers with shared-memory reloads. Measure register pressure and spills as well as latency; fewer source-level loads need not mean proportionally fewer DRAM reads. |
| Our loop address generation sometimes trails NVCC. | The matched [CUDA C++ reference](benchmark_rms.cu), inspected alongside the emitted PTX. | Open optimization. | In the 8192x64, eight-thread-row case, NVCC advances pointers while our code recomputes offsets. This is an observed instruction difference, not a proved attribution of the full timing gap. |
| Fresh processes cannot reuse our compiled artifacts. | [QuACK's source and runtime fingerprint, lines 56–86](https://github.com/Dao-AILab/quack/blob/14b0b7122e252a4df40a0b762c7ffd0f8b67a015/quack/cache/jit.py#L56) | In-memory cache only. | Add persistent artifacts keyed by source, compiler, target, specialization, and calling convention. Validate cache invalidation before reporting cold-start benefits. |

Cluster reductions, asynchronous tensor transfers, and tensor-core matrix
instructions are useful reference material, but the current small-row RMSNorm
and scalar transpose do not justify implementing all of them. Float16/bfloat16
and backward kernels remain separate milestones.

## Measured device performance

Measured on September 4, 2026 on a GH200, node `della-h23g2`, using CuTe DSL
4.7.1 and NVCC from CUDA 13.3. Both binaries use the same shapes, float32 input,
weights, scalar reduction algorithm, thread assignment, and CUDA stream.
NVCC uses `-O3 --fmad=true -arch=sm_90`; contraction is enabled to match the
fused multiply-add instructions observed in CuTe's output.

The reference is our matched scalar kernel, not a tuned NVIDIA or QuACK kernel.
The benchmark uses ordinary CUDA allocations and NVIDIA's
[CUDA Graph measurement helper](https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/testing.py#L511).
Each sample has 20 warmup launches and 200 measured launches; the table shows
the median of three samples. NVCC runs before CuTe in this final comparison.
Earlier calibration runs used the opposite order but different contraction
flags and are excluded from this table.

The same working buffers are reused, so these are warm working-set timings.
They exclude compilation, allocation, transfers, MLX execution overhead, and
per-launch Python dispatch. They do not establish HBM speed-of-light efficiency
or end-to-end model throughput. Both outputs were checked against a float64
NumPy reference before timing.

| Rows x width | Threads/row | Rows/block | Tiki/CuTe, µs | NVCC C++, µs |
| --- | ---: | ---: | ---: | ---: |
| 8192 x 16 | 32 | 4 | 2.275 | 2.267 |
| 8192 x 16 | 128 | 1 | 6.044 | 6.010 |
| 8192 x 16 | 8 | 16 | 1.811 | 1.759 |
| 8192 x 16 | 16 | 8 | 1.678 | 1.648 |
| 8192 x 64 | 32 | 4 | 2.636 | 2.577 |
| 8192 x 64 | 128 | 1 | 6.088 | 6.066 |
| 8192 x 64 | 8 | 16 | 4.003 | 3.272 |
| 8192 x 64 | 16 | 8 | 2.650 | 2.384 |
| 2048 x 4096 | 32 | 4 | 65.750 | 64.179 |
| 2048 x 4096 | 128 | 1 | 28.384 | 28.440 |

The narrowest case improves from 2.275 to 1.678 µs by assigning two rows to
each warp. The eight-thread setting regresses at width 64. NVIDIA's similar
thread-count heuristics accompany vectorized memory access and register
fragments, which our scalar implementation does not yet provide. We therefore
expose these choices explicitly and do not install a copied automatic heuristic.

All timing samples and numerical errors are retained in
[audit_results.json](audit_results.json). The largest recorded error is
3.5763e-7. The benchmark source is [benchmark_audit.py](benchmark_audit.py).

## Validation and reproduction

All 21 tests pass on the GH200. On the Mac, 14 capture/layout tests pass and
seven CUDA execution tests are skipped.

The new subgroup test fails before implementation. GPU correctness tests cover
8-, 16-, 32-, 64-, 128-, and 256-thread rows, partial row blocks, and odd widths.
Block sizes remain complete warps. Butterfly shuffle offsets stay below the
row-group width, so they cannot exchange values across row groups.
Filtered Compute Sanitizer synccheck reports zero errors for the generated
kernel. The independent unfiltered MLX allocation-ordering finding from
[COOPERATIVE_PROOF.md](COOPERATIVE_PROOF.md) remains unresolved.

On the GH200 with the existing environment:

```sh
python -m unittest discover -s experiments/cute_backend -p 'test_*.py'
python experiments/cute_backend/benchmark_audit.py \
  --subwarp --reverse --output /tmp/tiki-audit
compute-sanitizer --tool synccheck --kernel-name kne=tiki_fused --error-exitcode 1 \
  python -m unittest discover -s experiments/cute_backend -p test_cooperative.py
```

Use a fresh output directory. The benchmark requires CUDA Python, CuTe DSL,
NumPy, an importable MLX build, and `nvcc`. The CUDA C++ file is a benchmark
reference only; the Tiki compiler does not generate or compile CUDA C++.
