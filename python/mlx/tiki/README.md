# Reference layout layer

`mlx.tiki` combines a pinned PyCuTe algebra, Rust-owned indexing transforms,
and an MLX-backed reference Engine.
It is an experimental Python interface, not the Rust array backend or a
replacement for `mlx.core.array`.

## Representation

```text
tensor[coordinate] = engine[layout(coordinate)]
```

Layouts describe coordinates. `ArrayEngine` retains an MLX array and an integer
element offset. Scalar access uses MLX indexing and `.item()`, so it can
synchronize and is not a GPU kernel operation. It checks backing-array bounds
and does not reinterpret negative addresses as Python negative indices.

`from_array` explicitly normalizes a noncontiguous input, then constructs a
right-major reference tensor. This can allocate. `realize` performs no layout
conversion: it accepts integer-affine layouts, checks their addressed interval
against the Engine, and creates an `as_strided` view. Each shape leaf becomes an
MLX axis at this boundary. A manually constructed Engine must have a unit-stride
base for this operation.

XOR strides, coordinate-valued strides, and composed layouts cannot pass through
`realize`. Converting them to integer strides changes their meaning. Pure
reference algebra remains available for these maps, but no automatic nonlinear
kernel lowering is provided here.

## Composition and axis operations

`Swizzle` owns its validated parameters in Rust. Its immutable value accepts
nonnegative signed 64-bit indices. Generic composition and `.swizzle(...)`
share the same operation:

```python
import mlx.tiki as tk

base = tk.Layout((4, 4), (4, 1))
transform = tk.Swizzle(2, 0, 2)
layout = base.swizzle(transform)
assert layout == tk.compose(transform, base)
assert layout(1, 2) == transform(base(1, 2)) == 7
```

`ComposedLayout` retains an internal offset and supports scalar and partial
tensor indexing. Its slicing contract is:

```text
parent(fixed, free) = engine_delta + residual(free)
```

The fixed contribution stays inside a nonlinear composition when moving it to
the Engine would change addresses. The integer-displacement slicing interface
rejects XOR-valued displacements.

`unsqueeze`, `squeeze`, and `expand` operate on top-level modes. Nested modes
remain nested. Expansion uses zero strides and the trailing-axis compatibility
rule. Negative target extents are rejected.

## Scope and provenance

The reference accessors do not establish Rust ownership, exclusive mutation,
or kernel memory-safety proofs. In particular, the vendored foreign-pointer
accessors retain their upstream contracts. A pure layout's extended-coordinate
evaluation is not evidence that a tensor access is in bounds.

PyCuTe source remains unmodified. Its revision and license are recorded in
[_pycute/VENDORED.md](_pycute/VENDORED.md). Tiki's validation and MLX adapters
live outside that directory.

The Rust extension is required. Missing native bindings do not select a Python
implementation. See the [layout guide](../../../docs/src/usage/layouts.rst),
[recipes](../../../docs/src/examples/layouts.rst), and
[build instructions](../../../docs/src/dev/tiki_layouts.rst).

```sh
python -m unittest discover -s python/tests -p 'test_tiki_*.py'
```
