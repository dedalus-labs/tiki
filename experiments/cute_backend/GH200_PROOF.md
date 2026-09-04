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

The MLX source is MIT licensed. The `nvidia-cutlass-dsl` compiler package is
distributed under NVIDIA's CuTe DSL EULA, which remains a dependency constraint
for an internal backend.
