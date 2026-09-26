# Compiler contracts

The public entry points are `tiki.compile` and `associative_scan`. They accept
float32 MLX arrays. Compiler internals live in this package; import each contract
from the module that owns it.

```text
MLX export callback -> Captured -> specialize
                                  |-> Graph -> emitter -> Lowered
                                  |-> partition -> Program[Matmul | Collective | Kernel]
Lowered -> CudaIo -> cubin -> MLX CUDA kernel
Program -> cached MLX graph replay
```

| Modules | Responsibility |
| --- | --- |
| `events`, `capture` | Describe the native callback ABI and validate its signature, scalar constants, operation state, streams, and communicators. |
| `graph` | Own immutable tensor values, scalar constants, supported operation identities, and layout profiles. |
| `schedule`, `scan_schedule`, `lowered` | Validate launch geometry and derive kernel metadata. |
| `partition` | Form live CuTe regions at native operations, stream changes, and shape changes. |
| `program`, `native` | Build native stages and preserve collective order within each communicator. |
| `scalar`, `elementwise`, `row_reduction`, `transpose` | Emit pure MLIR for one supported kernel family. |
| `scan`, `scan_primitives` | Emit tile scans, lane/warp carries, and application of tile prefixes. |
| `scan_ad`, `scan_runtime` | Define derivative recurrences and execute recursive scans. |
| `arrays`, `specialize`, `compiled` | Normalize array results, own specialization caches, and register mathematical derivatives. |
| `pipeline`, `artifact`, `execution` | Specify NVIDIA's pipeline, own compilation I/O, and launch generated kernels. |

## Invariants

`Symbol` identifies a graph value; `GroupIndex` identifies a communicator within
one export. They are distinct types. `Profile` names shape and strides. Every
`Value` has a stride for each axis. Graphs have at least one output. Internal
graph and stage dataclasses are frozen and require named fields.

`Operation` and `CollectiveOperation` enumerate supported cases. Their dispatch
ends in `assert_never`, so adding an operation requires updating each exhaustive
consumer. A `Matmul` holds two inputs. A `Collective` holds one input, its group,
and its group index. Stages cannot contain an arbitrary Python execution callback.

The native event dictionaries exist only at capture. Every header has mandatory
fields and appears once. Primitive group fields are checked before constructing
a collective. Missing resources and unsupported operations raise errors during
lowering. The decoder never infers an operation from its display name alone.

Lowering produces text and metadata without a CUDA installation. `CudaIo` owns
NVIDIA compilation and temporary files. Its compiler import is deferred to this
boundary because the toolchain is Linux-only. The small `typings/cutlass` stub
describes the consumed 4.7.1 native bindings; `test_artifact.py` exercises them on
a GPU. Changing the dependency requires verifying this interface again.

Generated forward kernels are opaque inside the registered custom function.
Derivatives use the original primals and the mathematical graph. The low-level
`launch` method remains an explicit execution operation. A scan's forward and
derivative kernels use the same target architecture.

## Keep the boundary small

The checks follow Dedalus's
[Python guide](https://github.com/dedalus-labs/dedalus/blob/dev/docs/src/style/python.mdx):
Python 3.14, pinned Ruff and ty, complete annotations, and no `Any` or `object`
in compiler code. Compiler modules are capped at 300 lines, functions at 70,
control-flow nesting at three blocks, and signatures at five arguments.
Tests and demos also pass Ruff and ty. Negative type fixtures prove invalid
operation identities and missing communicator indices are rejected.

Build this checkout's native MLX extension first so its export API and generated
stubs agree. Then, from the repository root:

```bash
uv pip install -r experiments/cute_backend/requirements-check.txt
.venv/bin/python experiments/cute_backend/check.py
```

The CuTe Compiler workflow performs a CPU build and runs this command. GPU
contracts are exercised separately with the CUDA build and pinned CuTe dependency:

```bash
TIKI_TEST_ARCH=sm_89 PYTHONPATH=experiments/cute_backend \
  python -m unittest test_artifact test_associative_scan
```

Use `sm_90` for Hopper. Distributed execution additionally needs two ranks; see
[PROGRAMS.md](../PROGRAMS.md). Affine derivative oracles use sequential float64
recurrences independent of the GPU scan tree and MLX autodiff. Results and
emission hashes are in [LOWERING_VALIDATION.md](../LOWERING_VALIDATION.md).
