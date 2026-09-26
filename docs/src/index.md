---
html_theme.sidebar_secondary.remove: true
---

# Tiki

**Model code that reads like the math. Kernel code that says where every byte
goes. A compiler we can understand and steer.**

Tiki is Dedalus's machine learning framework. Write ordinary array code when
the computation is ordinary. When performance depends on a particular tile,
memory layout, or instruction, say so directly in Python. The same arrays,
automatic differentiation, and runtime surround both.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Layouts
:link: usage/layouts
:link-type: doc
CuTe-native layouts and index transforms on every array: a Tiki array is an
Engine paired with a first-class `Layout`.
:::

:::{grid-item-card} Compile
:link: tiki/compile/README
:link-type: doc
`tk.compile` lowers Tiki graphs to CuTe MLIR with explicit thread schedules,
consuming strided views in place.
:::

:::{grid-item-card} Associative scan
:link: tiki/scan/README
:link-type: doc
`associative_scan` with the interface of JAX, forward and reverse derivatives,
and kernels for any length.
:::

:::{grid-item-card} Rust runtime
:link: tiki/runtime/README
:link-type: doc
CUDA storage and completion owned by a checked Rust runtime behind a C++
boundary.
:::

:::{grid-item-card} Guide
:link: guide/index
:link-type: doc
The array framework underneath: lazy evaluation, unified memory, function
transformations, compilation.
:::

:::{grid-item-card} API
:link: api/index
:link-type: doc
`tiki.layout`, `tiki`, `tiki.nn`, and the C++ operations.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

Overview <tiki/index>
Guide <guide/index>
API <api/index>
Develop <develop/index>
```
