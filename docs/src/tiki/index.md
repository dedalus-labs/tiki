# Overview

What Tiki adds to the array framework, in the order the pieces depend on
each other: layouts on every array, a compiler that consumes them in place,
an associative scan built on that compiler, and a Rust runtime under it all.

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
CuTe layouts and index transforms as first-class values on every array.
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

```{toctree}
:caption: Records
:hidden:

Cooperative scheduling <compile/COOPERATIVE_PROOF>
Exemplar audit <compile/EXEMPLAR_AUDIT>
GH200 proof <compile/GH200_PROOF>
Runtime architecture <runtime/ARCHITECTURE>
ADR-0001 <runtime/DECISION-2026-09-05>
Allocator validation <runtime/VALIDATION-2026-09-06>
Backend reproductions <runtime/repros/README>
```
