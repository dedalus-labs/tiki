# Overview

Tiki adds three things to the array framework.

- Layouts. Every array carries a CuTe layout: a first-class map from
  coordinates to storage. The `tiki-cute` crate implements the layout algebra
  in Rust.
- Kernels. Kernels are written in `tk` from seven primitives, and the compiler
  proves their memory accesses from their layouts before it lowers them
  through LLVM's NVPTX backend.
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

:::{grid-item-card} Design
:link: ../dev/design
:link-type: doc
The nouns the framework is built from and the processors it targets.
:::

:::{grid-item-card} Kernels
:link: ../dev/kernels
:link-type: doc
The algebra kernels compute in, and every kernel built from the primitives.
:::

:::{grid-item-card} Runtime
:link: ../dev/runtime
:link-type: doc
Owned driver values, memory leased to in-flight work, and streams ordered
from the graph.
:::

::::

```{toctree}
:hidden:

Vision <vision>
Layouts <../usage/layouts>
Layout recipes <../examples/layouts>
Runtime <../dev/runtime>
```
