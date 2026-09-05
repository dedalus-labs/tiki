# CuTe compiler boundary

Current hardware results and diagnostic limitations are in
[COOPERATIVE_PROOF.md](COOPERATIVE_PROOF.md).

## Native MLX graph compilation

The experimental `tiki.py` module implements `tk.compile` for elementwise
graphs, one row sum with fused arithmetic, and tiled transpose. It uses MLX's
`export_function` callback to capture the native graph, constructs an explicit
thread schedule, emits CuTe MLIR directly, and passes it to `CuteCompiler`.
No reference CuTe kernel or generated Python source participates in this path.
MLX's C++ source is unchanged.

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
- The elementwise schedule supports add, subtract, multiply, negate, square,
  reciprocal square root, and scalar broadcasting. Its array inputs have the
  output shape or rank zero. Other schedules have the contracts below.
- MLX explicitly packs noncontiguous inputs to row-major buffers before
  launch; this can cost a copy.
- Threads per block: 32, 64, 128, or 256. Elements per thread: 1, 2, or 4.
  Element `i` belongs to `block * threads * elements_per_thread + thread +
  i * threads`. This is scalar work unrolling, not a promise of vector loads.
- Empty outputs launch no kernel. Nonempty launches use full thread blocks
  and guard each element, including the final partial tile.
- Graph and binary caches each retain up to 32 specializations in this process.
  Python globals and captured scalars are frozen when a shape is traced; pass
  changing values as array arguments. Persistent caching is not implemented.
- This prototype does not provide autodiff, float16/bfloat16, matrix
  multiplication, tensor-core schedules, or autotuning. The float16 normalization
  example in the root README remains a target; float32 works as shown below.

These are known float32 numerical semantics, subject to normal compiler
rounding and contraction. This is a correctness and integration experiment;
we have not measured its overhead against direct CuTe DSL.

## Cooperating on a row

```python
@tk.compile(schedule=tk.RowSchedule(threads_per_row=64, rows_per_block=2))
def rms_norm(x, weight):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6) * weight
```

The row schedule requires a 2D output and one sum over the last axis. It fuses
the supported arithmetic producing the sum and consuming its result. Inputs
can be scalars, column weights, row scalars, or full matrices. Reduction kind,
axis, and reciprocal-square-root semantics are validated from the exported
primitive parameters; this is not a match on the Python function's name.

`threads_per_row` selects 32, 64, 128, or 256 threads. `rows_per_block` selects
1, 2, 4, or 8 rows, subject to 256 total threads per block. Thread `t` owns
local row `t // threads_per_row` and columns starting at
`t % threads_per_row`, advancing by `threads_per_row`.

Each thread sums its elements, then uses warp shuffles to combine 32 partials.
If the row spans several warps, each warp's leader writes one value to shared
memory. All block threads reach the barrier, including threads assigned to an
unused row in the last block. Each warp then combines the shared partials.
For 64 threads per row and two rows per block, that scratch buffer is 16 bytes.
There is no shared scratch for a one-warp row.

MLX removes the sum when the width is one. The same row schedule accepts that
simplified graph and emits only its arithmetic. Zero rows are valid; zero
width and other reduction axes/kinds are rejected. This is a correctness-first
two-pass implementation: it rereads input values for the output computation.

## Controlling shared-memory banks

```python
@tk.compile(schedule=tk.TransposeSchedule(
    threads=128,
    swizzle=tk.Swizzle(bits=5, base=0, shift=5),
))
def transposed(x):
    return x.T
```

The transpose schedule stages a 32x32 float32 tile in shared memory. Threads
load rows, synchronize the block, and read columns to write contiguous output
rows. `threads` selects 32, 64, 128, or 256 threads per block. Partial input
and output tiles are guarded. The swizzle remains part of the CuTe type:

```text
!cute.memref<f32, smem, align<128>, "S<5,0,5> o 0 o (32,32):(32,1)">
```

`bits=0` selects the plain layout. For the 32x32 tile, the supported permutations
have `shift=5`, nonnegative `bits`/`base`, and `bits + base <= 5`. The public
`Swizzle.offset()` exposes the physical offset in float32 words:

```text
physical = logical XOR ((logical >> shift) AND (((1 << bits) - 1) << base))
```

In the plain layout, one warp reading 32 different words down a column hits a
single shared-memory bank. With `(5, 0, 5)`, those addresses occupy all 32 banks.
This is address accounting, not a claim of 32x speedup. This scalar-word layout
does not establish compatibility with Hopper's asynchronous tensor-copy or
matrix-instruction descriptors; those have additional layout constraints.

## Inspect the emitted program

```sh
python experiments/cute_backend/demo_cooperative.py --output /tmp/tiki-cooperative
# On the GH200, add --execute to save PTX/cubin and check numerical results.
python -m unittest discover -s experiments/cute_backend -p 'test_*.py'
```

Every demo case saves its graph, schedule, launch dimensions, shared-memory
size, and CuTe MLIR. With execution enabled it also saves PTX and the cubin.
`tk.binary(lowered).ptx` exposes the actual compiler output for inspection.

The optional layout inspector uses the monorepo's existing
`packages/python/tiki/src/tiki/kernels/cute/lib/debug.py` directly. Its
`visualize_tv` and `visualize_layout` tools landed in
[PR #2586](https://github.com/dedalus-labs/dedalus/pull/2586).
Run it in an environment with CuTe and matplotlib:

```sh
python experiments/cute_backend/inspect_layouts.py \
  --visualizer /path/to/dedalus/packages/python/tiki/src/tiki/kernels/cute/lib/debug.py \
  --output /tmp/tiki-layouts
```

The inspector checks every transpose thread/value coordinate against a real
CuTe layout before drawing. The visualizer uses column-major flattened
positions, so the supplied thread/value strides explicitly account for that
convention. Its bank diagrams show physical word offsets modulo 32; they do
not substitute for device-side race and bounds checks.

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
