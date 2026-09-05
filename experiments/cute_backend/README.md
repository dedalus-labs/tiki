# CuTe compiler boundary

## Native MLX graph compilation

The experimental `tiki.py` module now implements `tk.compile` for a small
elementwise subset. It uses MLX's `export_function` callback to capture the
native graph, constructs a thread schedule, emits CuTe MLIR directly, and
passes it to `CuteCompiler`. No reference CuTe kernel or generated Python
source participates in this path. MLX's C++ source is unchanged.

```python
# Run with experiments/cute_backend on PYTHONPATH.
import mlx.core as mx
import tiki as tk


@tk.compile(schedule=tk.Schedule(threads=128, elements_per_thread=4))
def affine(x: mx.array, y: mx.array) -> mx.array:
    return x * y + 2.0 - y


x = mx.arange(513, dtype=mx.float32)
y = mx.array(3.0)
lowered = affine.lower(x, y)
print(lowered.schedule)
print(lowered.mlir)
mx.eval(affine(x, y))  # Requires MLX CUDA on sm_90.
```

`lower()` runs on a Mac with MLX alone; it does not import the CuTe compiler or
execute the generated kernel. CUDA execution requires the CuTe dependency and
a CUDA-enabled MLX build. The provided demo writes the graph, schedule, and
MLIR to a fresh output directory:

```sh
python experiments/cute_backend/demo_compile.py --output /tmp/tiki-native
# On the GH200, add --execute and use a fresh output directory.
python -m unittest discover -s experiments/cute_backend -p test_compile.py
```

### Supported contract

- Pure positional array functions, one output, float32 only.
- Add, subtract, multiply, negate, square, and scalar broadcasting in the
  exported graph. Other primitives and non-scalar broadcasting are rejected.
- Array inputs have the output shape or rank zero. MLX explicitly packs
  noncontiguous inputs to row-major buffers before launch; this can cost a copy.
- Threads per block: 32, 64, 128, or 256. Elements per thread: 1, 2, or 4.
  Element `i` belongs to `block * threads * elements_per_thread + thread +
  i * threads`. This is scalar work unrolling, not a promise of vector loads.
- Empty outputs launch no kernel. Nonempty launches use full thread blocks
  and guard each element, including the final partial tile.
- Graph and binary caches each retain up to 32 specializations in this process.
  Python globals and captured scalars are frozen when a shape is traced; pass
  changing values as array arguments. Persistent caching is not implemented.
- This prototype does not provide autodiff, reductions, matrix multiplication,
  tensor-core schedules, swizzles, or autotuning. The `tk.compile` normalization
  example in the root README remains a target.

These are known float32 numerical semantics, subject to normal compiler
rounding and contraction. This is a correctness and integration experiment;
we have not measured its overhead against direct CuTe DSL.

## Original artifact-boundary probe

This experiment tests whether Tiki can generate CuTe MLIR directly and pass it
to NVIDIA's compiler without generating or importing temporary Python modules.

CuTe DSL has three separate responsibilities:

1. Its Python decorators rewrite and trace kernel source into a
   `PreCompiledMlirArtifact`.
2. `CuteCompiler` lowers that artifact to an object containing CUDA device
   code and a host launcher.
3. The CUDA runtime loads and executes the device code.

Tiki intends to replace the first responsibility for supported MLX graph
regions. The reference source kernel in `probe.py` exists only to produce a
known-good artifact against which a future MLX-to-CuTe-MLIR emitter can be
compared.

```text
MLX region -> Tiki schedule -> CuTe MLIR -> CuteCompiler -> cubin -> MLX runtime
```

The experiment establishes the boundary at serialized and textual CuTe MLIR.
Both forms compile to a host object and a cubin. It does not claim that MLX
currently contains enough scheduling information to generate a fast kernel.
Scheduling, layout selection, and autotuning remain explicit compiler
responsibilities.

## Run

Use Linux, Python 3.13, an NVIDIA GPU, and CUDA 12.9 or newer.

```sh
python -m pip install -r experiments/cute_backend/requirements.txt
python experiments/cute_backend/probe.py --arch sm_90 --output /tmp/tiki-cute
python experiments/cute_backend/run_mlx.py --arch sm_90
```

The probe performs two compilation stages. It serializes the pre-pass MLIR,
then starts `compile_mlir.py` in a separate process. That process does not
import the kernel source or `cutlass.cute`; it deserializes the artifact and
asks `CuteCompiler` to produce an object and cubin. No generated Python file is
used after the MLIR stage.

The probe also reparses the textual MLIR with no CuTe function metadata and
compiles it with the undecided ABI. This is the path a native MLX emitter can
use when it owns kernel launch and does not need CuTe's host wrapper.

`run_mlx.py` launches the cubin emitted from the serialized MLIR through MLX's
`precompiled_cuda_kernel` primitive. This tests buffer ownership, argument
ordering, stream integration, and execution without using DLPack or CuTe's host
executor.

The native graph path above replaces that reference frontend for the supported
elementwise subset. The original probe remains an independent reference for
the compiler artifact interface.
