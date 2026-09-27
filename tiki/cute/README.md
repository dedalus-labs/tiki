# tiki-cute

tiki-cute is Tiki's implementation of the CuTe layout algebra. A layout maps
the coordinates of a tensor to memory offsets. The algebra derives every
layout a kernel needs, such as tiles, thread partitions and tile indices, from
the layouts the kernel is given. Tiki's kernel compiler builds on it to prove
that a kernel's addresses stay inside its buffers.

The algebra follows [Cecka's formalization of CuTe][cecka] and agrees with
[PyCuTe][pycute], the reference implementation, on every integer and `F2` case
in PyCuTe's own test suite. Extents may be dynamic, known only at launch, and every operation on
them either proves its preconditions or refuses.

## A layout and its tiles

A 4 by 4 row-major matrix is the layout `(4, 4):(4, 1)`. Element `(r, c)`
lives at offset `4 * r + 1 * c`. A kernel that loads 2 by 2 tiles needs each
axis split into the position within a tile and the tile index. Deriving those
strides by hand is error-prone, and a mistake produces a wrong address rather
than an error. `logical_divide` derives them:

```rust
use tiki_cute::{Layout, Tiler};

let matrix: Layout = "(4, 4):(4, 1)".parse()?;
let tiles = matrix.logical_divide(&"(2, 2)".parse::<Tiler>()?)?;
assert_eq!(tiles.cute().to_string(), "((2, 2), (2, 2)):((4, 8), (1, 2))");
println!("{}", tiles.describe());
# Ok::<(), tiki_cute::LayoutError>(())
```

`describe` prints one row per coordinate with its extent and stride:

```text
coordinate  extent  stride
c[0][0]     2       4
c[0][1]     2       8
c[1][0]     2       1
c[1][1]     2       2
index = 4 * c[0][0] + 8 * c[0][1] + 1 * c[1][0] + 2 * c[1][1]
```

`c[0][0]` is the row within a tile and `c[0][1]` the tile row. The offsets
have not moved. Only the coordinates that name them have.

## Extents known at launch

A kernel that tiles an `M` by `K` matrix is compiled once for many values of
`M`. The compiler still has to compute with `M`. The tile count along the rows
is `ceil(M / 128)`, and whether a tile of 128 rows fits inside a mode of the
layout depends on `M`.

tiki-cute represents such an extent as a polynomial over launch parameters
([`Param`]). A parameter carries facts the launcher checks before the kernel
runs: it is positive, and it may be a multiple of a divisor. The algebra
computes with the polynomial and decides each precondition from those facts:

```rust
use tiki_cute::{Int, Layout, Param, Shape, Tiler, Tuple};

let rows = Int::from(Param::multiple_of("M", 128)?);
let column = Layout::from(Shape::try_from(Tuple::Leaf(rows))?);
let tiled = column.logical_divide(&Tiler::try_from(128)?)?;
assert_eq!(tiled.cute().to_string(), "(128, floor(M/128)):(1, 128)");
# Ok::<(), tiki_cute::LayoutError>(())
```

Because `M` is a multiple of 128, `ceil(M / 128)` is exactly `floor(M / 128)`.
Every question the facts cannot settle comes back as [`Truth::Open`], and an
operation that needs the answer refuses with [`Verdict::Unproven`]. It never
guesses. [ARCHITECTURE.md](ARCHITECTURE.md) explains the representation and
each decision in detail.

## Swizzles as XOR strides

A kernel that stages a tile in shared memory swizzles its offsets, so that the
threads of a warp reach different banks. `Swizzle(bits=3, base=0, shift=3)`
XORs bits 3 to 5 of an offset into bits 0 to 2. The same function is a layout
whose strides add by XOR, and as a layout it passes through the whole algebra:

```rust
use tiki_cute::{Layout, Swizzle, SwizzleParams, Tiler};

let swizzled: Layout = "(8, 8):(^1, ^9)".parse()?;
let swizzle = Swizzle::try_from(SwizzleParams { bits: 3, base: 0, shift: 3 })?;
for index in 0..64 {
    let offset = swizzled.at(index).as_xor().map_or(0, |bits| bits.value());
    assert_eq!(offset, swizzle.apply(index)?);
}
let tiles = swizzled.zipped_divide(&"(2, 4)".parse::<Tiler>()?)?;
assert_eq!(tiles.cute().to_string(), "((2, 4), (4, 2)):((^1, ^9), (^2, ^36))");
# Ok::<(), tiki_cute::LayoutError>(())
```

`^9` is an XOR stride, a value in F2 as [`Xor`] defines it: offsets add by
XOR, and an integer multiplies a stride carry-lessly, so element `(r, c)` lives
at `r ^ 9c`. The strides of one layout are all integers, all arithmetic tuples
or all XOR values, and a layout with XOR strides has static extents, because a
carry-less product needs every bit of the integer it multiplies.

The algebra's walks multiply and add integers, which agree with their
carry-less counterparts only where no bit carries. Each XOR result that relies
on that is checked at every index, which static extents make exact.

## Where tiki-cute differs from PyCuTe

- Composition refuses a hierarchical right layout whose modes' offsets can sum
  across a mode of the left layout. Cecka states the condition as Equation 23
  of [the specification][cecka]. PyCuTe skips it, so for `(4, 3):(0, 1)` composed with
  `(2, 3):(2, 1)` it returns `(2, 3):(0, 0)` where the fifth offset should be 1.
- A symbolic condition is decided from parameter facts. PyCuTe decides it by
  sympy's structural comparison, which can accept a condition that fails for
  some launches.
- XOR strides follow PyCuTe's `F2` paths, except where PyCuTe returns a layout
  that is not the operation's result. PyCuTe composes `16:^1` with `4:3` as
  `4:^3`, which maps 3 to `3 * ^3 = ^5` where `16:^1` maps `3 * 3` to `^9`.
  tiki-cute refuses such a composition or layout sum, cuts a right inverse to
  the whole modes that invert, refuses a left inverse that fails its contract,
  and refuses a product whose copies lie in another codomain. PyCuTe keeps the
  XOR zero `F0` apart from 0, and tiki-cute reads `^0` as the integer 0.
- A tiler holes individual modes with `_`. A whole-layout hole, PyCuTe's
  `None` tiler, is spelled by not calling the operation. `zipped_divide`
  gathers every mode's tile into one mode, so it needs a layout for every mode.

## Printing

Layouts print in the forms of the `tiki.layout` Python API.
`Layout(shape=(4, 4), stride=(4, 1))` is the default form.
[`Layout::cute`] gives CuTe's `(4, 4):(4, 1)`, and [`Layout::describe`] gives
the coordinate table above. CuTe notation also parses, and every printed
layout parses back to the same layout, so the reference cases and this crate's
tests are written as CuTe prints them. The notation carries no parameter
facts: a name parses as a parameter that admits every positive integer. An XOR
stride prints as `^9`, where PyCuTe prints `F9`, so no name reads as one.

## Verification

- `tests/reference.rs` replays every integer and `F2` layout-algebra call that
  PyCuTe's test suite makes and requires the same result or the same refusal.
  `tests/reference/record.py` regenerates the cases from a PyCuTe checkout.
- `tests/postconditions.rs` checks each operation's contract on random layouts
  by evaluating every index.
- `tests/symbolic.rs` checks that every symbolic result satisfies the same
  contracts after concrete values replace its parameters.
- `tests/tensor.rs` checks that a tensor accepts exactly the placements whose
  bytes lie inside its bounds.
- `tests/xor.rs` checks each operation's contract on random XOR layouts, that a
  [`Swizzle`] and its XOR layout agree at every index, and each difference
  from PyCuTe's `F2` paths.

## Build

tiki-cute needs Rust 1.92 or newer. Python source builds compile it on every
backend, CPU and Metal included, because the `tiki.layout` Python API computes
every layout operation with it. The workspace member `python/` is the PyO3
crate `tiki-cute-python`, which builds the `tiki.layout._cute` extension:
`CMakeLists.txt` runs Cargo with its `extension-module` feature and installs
the library next to the `tiki.layout` package. The binding forbids unsafe code
and passes typed values only, so Python never formats CuTe notation for this
crate to parse. The crate's semantics do not depend on the device. Prebuilt
wheels need no Rust toolchain, and C++ builds with
`TIKI_BUILD_PYTHON_BINDINGS=OFF` skip the crate. The CUDA runtime has its own
Rust crate.

```sh
cargo test --manifest-path tiki/cute/Cargo.toml --workspace
cargo clippy --manifest-path tiki/cute/Cargo.toml --workspace --all-targets -- -D warnings
cargo fmt --manifest-path tiki/cute/Cargo.toml --all --check
```

The binding's tests link libpython. PyO3 finds the interpreter on `PATH`, or
the one `PYO3_PYTHON` names.

## References

1. Cris Cecka. [CuTe Layout Representation and Algebra](https://arxiv.org/abs/2603.02298).
   arXiv:2603.02298, 2026. The specification this crate implements. Equation
   and section numbers in the source refer to it.
2. Jack Carlisle, Jay Shah, Reuben Stern and Paul VanKoughnett.
   [Categorical Foundations for CuTe Layouts](https://arxiv.org/abs/2601.05972).
   arXiv:2601.05972, Colfax Research, 2026, with its companion
   [layout-categories](https://github.com/ColfaxResearch/layout-categories/tree/689369f8b9c61cbcf9aa81c20c83f0ad7cef9719).
   A second formal account of the algebra, through the categories `Tuple` and
   `Nest`, consulted as inspiration. No code is taken from it.
3. NVIDIA. [PyCuTe](https://github.com/NVlabs/CuTe/tree/111253d17e2f0f8631f43999b43ac4afa5954b04),
   the Python reference implementation of the algebra, Apache-2.0. The
   reference cases are recorded from this commit.
4. NVIDIA. [CuTe in CUTLASS](https://github.com/NVIDIA/cutlass/tree/0b55a2f691d69981583568fd9eb69687b1f0de8a/include/cute),
   the C++ implementation, and the CuTe DSL's
   [`cute.assume`](https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/python/CuTeDSL/cutlass/cute/core.py#L2260-L2267),
   which attaches a divisibility fact to a dynamic integer as [`Param`] does.
5. NVIDIA. [cuda-oxide](https://github.com/NVlabs/cuda-oxide/tree/ec4aa4797956534578a1af010f86252a0b6d8626),
   Rust kernels compiled to PTX, and
   [cuTile Rust](https://github.com/NVlabs/cutile-rs/tree/cc720f182f38bf46527753caa340e78d6d5fa3cc)
   0.4.0. ADR-0001 evaluates both, and neither is a dependency.
6. NVIDIA. [CUDA Tile IR](https://github.com/NVIDIA/cuda-tile/tree/7e8e2e68fa219716103824c01f7303367cf7df8d),
   evaluated as a kernel target. Tile IR has no threads or shared memory, which
   CuTe-style kernels address explicitly.

[cecka]: https://arxiv.org/abs/2603.02298
[pycute]: https://github.com/NVlabs/CuTe/tree/111253d17e2f0f8631f43999b43ac4afa5954b04
