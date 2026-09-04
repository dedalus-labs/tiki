# CuTe compiler boundary

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

The next experiment must replace the reference decorator frontend with a small
MLX region emitter and compare the resulting operation semantics before adding
performance work.
