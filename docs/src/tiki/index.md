# Overview

Tiki adds four things to the array framework. Each one depends on the one
before it.

- Layouts. Every array carries a CuTe layout: a first-class map from
  coordinates to storage.
- A compiler. `compile` lowers a graph to CuTe MLIR and consumes those layouts
  in place.
- An associative scan. The scan is built on the compiler and has forward and
  reverse derivatives for any length.
- A Rust runtime. Storage and completion on CUDA are owned by a checked Rust
  runtime behind a C++ boundary.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Vision
:link: vision
:link-type: doc
Why Tiki exists, what works today, and the API it is working toward.
:::

:::{grid-item-card} Layouts
:link: ../usage/layouts
:link-type: doc
CuTe layouts and index transforms as values on every array.
:::

:::{grid-item-card} Layout recipes
:link: ../examples/layouts
:link-type: doc
Tiling, broadcasting, negative strides, nested composition, and the errors.
:::

:::{grid-item-card} Compiler
:link: compile/README
:link-type: doc
`compile` lowers graphs to CuTe MLIR with explicit thread schedules.
:::

:::{grid-item-card} Associative scan
:link: scan/README
:link-type: doc
The generic tree and the CUDA kernels, with derivatives checked against each other.
:::

:::{grid-item-card} Rust runtime
:link: runtime/README
:link-type: doc
CUDA storage and completion owned by a checked Rust runtime.
:::

::::

```{toctree}
:hidden:

Vision <vision>
Layouts <../usage/layouts>
Layout recipes <../examples/layouts>
Compiler <compile/README>
Associative scan <scan/README>
Rust runtime <runtime/README>
```
