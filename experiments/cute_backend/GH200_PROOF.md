# GH200 proof

Validated on 2026-09-04 against MLX commit
`b6368984b8e02a3fb3ee7986846c0fb85e1fccf7`.

## Environment

- NVIDIA GH200 144 GB HBM3e, compute capability 9.0
- CUDA Toolkit 13.3
- Python 3.13
- `nvidia-cutlass-dsl==4.7.1`
- MLX built from this tree with its CUDA backend enabled

NCCL was absent from this allocation. The build used MLX's no-NCCL path because
this experiment does not exercise distributed execution.

## Compiler boundary

`probe.py` compiled one vector-add specialization through both supported inputs:

| Input | Host object | Cubin | SHA-256 |
| --- | ---: | ---: | --- |
| Serialized `PreCompiledMlirArtifact` | 8,728 bytes | 4,416 bytes | `9e283288c5ea86dcde85704c6281714cc437241dae59d5cf4e9fb91b8a017b67` |
| Raw textual CuTe MLIR | 8,704 bytes | 4,416 bytes | `9e283288c5ea86dcde85704c6281714cc437241dae59d5cf4e9fb91b8a017b67` |

The cubins are byte-identical. Raw textual MLIR has no CuTe function metadata,
so it compiles with `Abi.Tbd`; the serialized artifact retains the generated
CutlassCall host-function metadata. Both contain the same device kernel.

## MLX launch

`run_mlx.py` loaded the cubin produced from serialized MLIR with
`mx.fast.precompiled_cuda_kernel`, passed two MLX-owned input buffers and one
MLX-owned output buffer, and evaluated 4,096 float32 additions on the MLX CUDA
stream. Maximum error was exactly `0.0`.

This proves that Tiki can own an MLX-to-CuTe-MLIR frontend and use NVIDIA's
compiler as an artifact backend without generating Python source or using
CuTe's host executor. It does not yet prove a scheduler, layout system,
autotuner, graph partitioner, or a performance advantage.

## Native graph follow-up

The first `tk.compile` slice was validated on the same GH200 on 2026-09-04.
It traces `x * y + 2.0 - y` with MLX's native export callback and emits CuTe
MLIR directly. The arithmetic emitter does not call a CuTe-decorated reference
function. It compiles the emitted text with `CuteCompiler` and executes through
MLX's existing CUDA primitive. No MLX C++ changes were needed.

The demo used 513 float32 elements, a scalar multiplier, 128 threads per block,
and four elements per thread. The launch had two blocks, with a guard for every
element. Maximum error was `0.0`.

All ten tests in `test_compile.py` passed on the GH200. They cover native graph
capture, schedule mapping, unsupported operations/dtypes/broadcasts,
data-dependent Python rejection, shape specialization, float arithmetic,
scalar outputs, noncontiguous input packing, empty and partial tiles, and
current input values on cached execution. On the Mac, seven capture tests
passed and three CUDA execution tests were skipped. The Mac also emitted the
demo graph, schedule, and MLIR without importing CuTe.

This adds a fixed elementwise schedule to the original proof. It does not yet
establish autotuning, reductions, shared-memory schedules, differentiation, or
the compile-time/performance budgets in the root README.

The MLX source is MIT licensed. The `nvidia-cutlass-dsl` compiler package is
distributed under NVIDIA's CuTe DSL EULA, which remains a dependency constraint
for an internal backend.
