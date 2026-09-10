# Matrix multiplication and collectives in compiled programs

`tk.compile` now lowers a float32 function into fused CuTe arithmetic regions and explicit native MLX operations. Matrix multiplication uses MLX's existing CUDA implementation. All-reduce, all-gather, and sum-scatter use the captured communication group. Their input and output dependencies remain in the lazy MLX graph, so CUDA can overlap independent work.

This is available in the modified Tiki checkout. The experimental `tiki` module lives in `experiments/cute_backend`; it is separate from the layout API in `mlx.tiki`. The code requires a CUDA build of this checkout and CuTe DSL 4.7.1. The accepted kernel targets are L4 (`sm_89`) and Hopper (`sm_90`).

```python
import mlx.core as mx
import tiki as tk

group = mx.distributed.init(strict=True, backend="nccl")
schedule = tk.Schedule(arch=mx.device_info(mx.gpu)["architecture"])

@tk.compile(schedule=schedule)
def step(a, b, x):
    return a @ b, mx.distributed.all_sum(x * 0.5, group=group) + 1.0

x = mx.full((1024, 1024), group.rank() + 1, dtype=mx.float32)
product, reduced = step(x, mx.ones_like(x), x)
mx.eval(product, reduced)
```

The matrix multiplication and communication branch are independent. The input scaling and output addition compile into CuTe kernels on either side of NCCL. There is no CPU wait inside `step`; `mx.eval` waits when the caller wants both results. A single MLX stream can already expose overlap through CUDA graph dependencies, so the example does not add an unnecessary second stream. This does not guarantee a speedup for every shape or machine.

The [complete example](demo_overlap.py) includes two-rank checks and a numerical oracle. From the repository root, with the environment containing the CUDA build activated:

```bash
PYTHONPATH=python:experiments/cute_backend \
  mlx.launch --backend nccl -n 2 -- python experiments/cute_backend/demo_overlap.py
```

The launcher selects GPUs 0 and 1 and sets the NCCL rank and rendezvous environment. It requires two NVIDIA GPUs. A Mac can run the local graph-lowering tests, but it cannot execute these CuTe CUDA kernels.

Call `step.lower(a, b, x)` to inspect its `Program`. Its `stages` show `Matmul`, `CuTe`, `AllReduce`, and `CuTe`. Each CuTe stage exposes its `lowered.mlir`; native stages retain their operation, stream, and communication-group index. Unsupported primitives fail during lowering. They do not select another backend.

The first supported programs use the elementwise `Schedule`, float32 arrays, and positional arguments. They can return one array or a flat tuple. Local sum reductions and tiled transposes retain their separate cooperative-schedule contracts. Automatic row chunking, CuTe-generated matrix multiplication, and GPU-initiated network kernels are outside this change. Mixed-program differentiation is limited by the operations supported when lowering the derivative graph; unsupported derivative primitives raise an error.

## Ownership and ordering

The C++ export callback now retains each primitive's MLX stream. Collective events additionally carry the actual `Group` and a group index scoped to that export. The reduction kind remains explicit. A collective cannot be mistaken for a local reduction with a similar printed name, and binary file export rejects unsupported collective resources.

Python partitions the exported tape at native operations, stream changes, and incompatible arithmetic shapes. It compiles each supported arithmetic region, then constructs native MLX nodes around those kernels. The resulting program uses MLX graph replay, so warm calls do not walk the stage list in Python. Successive collectives in one captured group receive an explicit dependency, preserving their tape order even on distinct streams. Collectives in separate groups keep their distinct identities. Captured Python values and groups are fixed for a specialization; create a new compiled function when those captures change.

MLX retains arrays through device completion. This change continues to use that mechanism and leaves the Rust allocator unchanged. Registered NCCL windows for GPU-initiated networking would require a separate allocation and lifetime contract, following NVIDIA’s [window-registration requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html).

## Implementation language

Python is appropriate for tracing, graph partitioning, specialization, and construction of compiler input at this stage. NVIDIA's [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html) uses Python to build compiler input, and PyTorch's [Inductor lowerings](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/lowering.py) use Python to construct their intermediate representation. The generated GPU kernels execute natively, and the assembled program is cached for native graph replay.

C++ is the current native boundary to MLX and the natural fit for passes implemented directly within MLIR. MLIR's [Python bindings](https://mlir.llvm.org/docs/Bindings/Python/) expose native compiler structures through a C interface, so a native pass can be introduced without rewriting the user-facing Python interface. Rust remains the owner of CUDA allocation and resource lifetime. Port a lowering pass only when profiling identifies compile time or Python dispatch as a meaningful cost; no such comparison has established a need for a language migration here.

## Validation

The export regression failed before the C++ fix: the callback reported `Reduce` for an all-reduce, lost its reduction kind, and binary export could write the wrong primitive. The corrected export tests exercise collective identity, state, stream, and rejection of unsupported file serialization.

Local tests cover graph partitioning, stream-sensitive caching, explicit stream placement, independent outputs, and unsupported operations. A real two-process ring group checks all-reduce sum/min/max, all-gather, and ordering across streams. Ring does not implement sum-scatter, so that execution test requires NCCL.

```bash
PYTHONPATH=python python -m unittest discover -s experiments/cute_backend -p 'test_*.py'
TIKI_TEST_BACKEND=ring PYTHONPATH=python:experiments/cute_backend \
  mlx.launch --backend ring -n 2 -- python experiments/cute_backend/test_native_program.py
TIKI_TEST_BACKEND=nccl PYTHONPATH=python:experiments/cute_backend \
  mlx.launch --backend nccl -n 2 -- python experiments/cute_backend/test_native_program.py
```

On September 9, 2026, the built package passed all 11 program tests, all 5 NCCL tests, and the complete demo on each of two AWS L4 nodes. The checks include generated arithmetic around matrix multiplication and communication, strided inputs, partial tiles, separate streams, sum-scatter, and distinct groups. Local validation also passed 10 C++ export cases, 21 Python export cases, 9 device/stream cases, and 30 compiler cases; GPU-only and unlaunched distributed cases were skipped locally. The two-node checks used NCCL over the network and did not exercise GPU-initiated networking.

A matched two-node probe used a 4096×4096 float32 matrix multiplication and a 4 MiB all-reduce branch with scaling and addition. Seven samples of five completed evaluations, taking the slower rank per sample, gave medians of 4.814 ms with an explicit wait, 4.383 ms for native MLX, and 4.439 ms for Tiki. A separate Nsight Systems trace showed overlapping matrix-multiplication and NCCL kernel intervals in all 35 Tiki evaluations on each GPU; the explicit-wait control had none. This demonstrates preserved overlap, not a performance advantage over native MLX. A preceding controlled probe reduced the Tiki median from about 4.56 to 4.42 ms by caching native graph replay instead of executing the Python stage loop on each call.
