# tiki-cute architecture

tiki-cute is the layout layer of Tiki's kernel compiler. Every address a Tiki
kernel computes is an offset of a layout built with this algebra. The
compiler's proofs of bounds and disjoint writes are therefore proofs about
these layouts, and the algebra must be exactly right on every layout it
returns.

This document explains how the crate represents layouts, how it computes with
extents known only at launch, and each decision where the correct behavior was
not obvious. The [README](README.md) introduces the API and lists the
references.

## Where the algebra runs

ADR-0001 fixes Tiki's kernel pipeline. A kernel written with `@tk.kernel`, or
chosen by the compiler for a fused Tiki region, is traced into kernel IR. The
kernel IR lowers to a thread IR, then to LLVM IR, which LLVM's NVPTX backend
compiles to PTX in a separate compiler process.

The layout algebra runs while the kernel is traced, inside the compiler.
Layouts are values of ordinary Rust types, built and transformed by ordinary
code. CUTLASS takes the other route and encodes static layouts in C++ template
types, so every new layout instantiates new templates. Keeping layouts as
values makes a kernel's layout work cost what the algebra costs, before LLVM
runs.

## Representation

### One recursive tuple

Cecka builds every part of a layout from hierarchical tuples: a shape is a
tuple of extents, a stride a tuple of offsets, a coordinate a tuple of
integers. [`Tuple<T>`](src/tuple.rs) is that one structure, a leaf or a
sequence of tuples. Every algorithm in the crate is recursion over it, and the
two relations the algebra needs are its methods:

- `congruent`: two tuples have the same profile, leaf for leaf.
- `zip`: pairs each leaf of one tuple with the subtree of another in the same
  position. It is defined exactly when the first tuple is weakly congruent to
  the second, so it both checks and uses weak congruence.

### Shapes, strides and layouts

[`Shape`](src/shape.rs) wraps a tuple of extents and guarantees each extent is
positive for every launch. [`Stride`](src/stride.rs) wraps a tuple of offsets
and guarantees they lie in one codomain. A [`Layout`](src/layout.rs) pairs a
shape with a congruent stride.

Every public constructor checks these invariants, and the algebra builds new
layouts only from checked ones. Code that holds a `Layout` never tests them
again.

### Integers known now or at launch

An extent or stride is an [`Int`](src/int.rs). `Int::Static` is an `i64` the
compiler folds. `Int::Dynamic` is a polynomial over launch parameters with
integer coefficients, such as `4*N` or `2*N + 2`.

A polynomial can also contain `floor(p / q)`, an opaque quotient, where no
exact division exists. The tile count `ceil(M / 128)` is such a quotient,
written `floor((M + 127) / 128)`.

[`Poly`](src/poly.rs) keeps every polynomial in a canonical form: a sorted map
from monomials to nonzero coefficients. A constant polynomial is always
`Int::Static`. Two structurally equal `Int`s are therefore equal for every
launch, and the algebra proves an equality by comparing structure.

The converse does not hold. `floor(floor(M / 2) / 2)` and `floor(M / 4)` agree
for every `M` and differ structurally. The algebra therefore acts on structural
equality only where equality is what it needs proven, such as merging two
modes, and never reads structural inequality as proof that two values differ.

Arithmetic takes the `i64` path when both operands are static. Almost every
integer in a kernel's tile layouts is static, so the polynomial path runs only
for the few layouts that carry launch parameters.

### Three codomains

A memory layout maps coordinates into Z. A coordinate layout maps them into
Z^n, and each stride is a scaled basis element such as `2@1`, twice the second
unit vector. A kernel names positions with coordinate layouts: an identity
tensor for boundary masks, a TMA copy's coordinates, or a tensor-memory address
of a lane and a column. A swizzled layout maps them into F2, where offsets add
by XOR. [`Xor`](src/xor.rs) defines that codomain, and the section on XOR
strides below explains how the algebra treats it.

[`Offset`](src/offset.rs) is one type for all three. An integer is the rank-0
case of an arithmetic tuple, and an XOR value is its own variant. The value is
canonical, with trailing zero components trimmed, and an all-zero tuple and the
XOR value `^0` equal to the integer 0. Zero therefore belongs to every
codomain, and equal offsets are structurally equal.

A nonzero integer, an arithmetic tuple and an XOR value have no sum with one
another. `Stride::try_from` refuses a stride that mixes them, so every sum the
algebra forms is defined.

## Static and dynamic extents

A kernel that tiles an `M` by `K` matrix is compiled once and launched for
many values of `M`. Compiling once per value of `M` would be correct, and it
is what a compiler that only folds constants must do. It costs one compilation
for every shape a model runs.

To compile once, the algebra must compute with `M` itself. Arithmetic is easy
with polynomials. The hard part is the questions: whether 128 divides `M`,
whether a tile of 128 rows fits inside a mode of `M` rows, whether one stride
lies past another. Their answers can depend on `M`.

PyCuTe computes with sympy symbols and answers these questions by sympy's
structural comparisons. Its complement, for instance, checks that two modes do
not overlap only when both strides are static, so a symbolic complement can be
wrong for some launches. Tiki's proofs cannot rest on a result that holds only
for some launches.

### Facts

A launch parameter is a [`Param`](src/param.rs): a name and facts about every
value it may take. Every parameter is positive, and it may be a multiple of a
divisor:

```rust
use tiki_cute::Param;

let rows = Param::multiple_of("M", 128)?;
assert!(rows.admits(256));
assert!(!rows.admits(200));
# Ok::<(), tiki_cute::LayoutError>(())
```

The facts are a contract with the launcher. The compiler may rely on them.
ADR-0001 has the runtime check each launch against the kernel's contract, and
`Param::admits` is that check for one argument, so a launch that breaks a fact
is refused before any address exists. CuTe DSL's
`cute.assume(value, divby=…)` attaches the same kind of fact to a dynamic
integer.

### Truth

Each predicate on an `Int` returns a [`Truth`](src/truth.rs):

| Answer | Meaning |
| --- | --- |
| `Proven` | holds for every launch the facts admit |
| `Refuted` | fails for every admitted launch |
| `Open` | depends on the launch, or the facts are too weak to tell |

The answers come from what the facts prove about the polynomial:

- A parameter is at least its divisor, so each polynomial has a lower bound,
  and an upper bound when every non-constant coefficient is negative.
  `is_positive`, `is_at_least` and `equals` compare these bounds.
- A term `c * M * N` is a multiple of `c` times the divisors of `M` and `N`.
  `is_multiple_of` proves a static divisor from these known multiples.
- A dynamic divisor is proven only by exact polynomial division, so `N*M` is a
  multiple of `N`.

Floor division uses the same facts. When 128 divides `M`, the dividend of
`floor((M + 127) / 128)` splits into `M`, which 128 divides, and `127`, which
lies in `0..128` for every launch. The remainder adds nothing to the floor, so
`ceil(M / 128)` and `floor(M / 128)` produce the same canonical atom.

### What an open answer means at each call site

`Open` is a third answer, and each call site in the algebra states what it
means there. An optional rewrite runs only when its condition is proven,
because skipping it leaves a correct layout that is less simplified. A
precondition calls `Truth::require`, which returns a
[`LayoutError::Condition`](src/error.rs) naming the condition, with
`Verdict::Violated` for a refuted one and `Verdict::Unproven` for an open one.
The kernel author can then add the fact that proves it.

| Operation | Question | An open answer |
| --- | --- | --- |
| coalesce | do two adjacent modes merge? | keeps them apart |
| coalesce | is an extent 1? | keeps the mode |
| compose | does the stride skip whole leading modes? | stops skipping, and the next check decides |
| compose | does the stride split the first mode exactly? | refuses: stride divisibility |
| compose | does the rest fit inside the next mode? | uses the mode in full, and requires shape divisibility |
| compose | do summed modes have segregated images? | refuses: segregated images |
| complement | does each mode start past the chain below it? | refuses: injectivity |
| left inverse | is each stride a multiple of the chain below it? | refuses: ordered chain |
| recast | does the stride or the factor divide the other? | refuses: recast divisibility |
| layout addition | is there a common domain? | refuses: common domain, whenever an extent is dynamic |

Several algorithms sort modes by stride. Static strides sort by value and
dynamic ones follow in their original order, as in PyCuTe. The sort itself
decides nothing. Each step that depends on the order is proven, so an order
the facts cannot confirm ends in a refusal.

### A running example

A padded matrix has `M` rows and a leading dimension of 4096 elements, the
layout `(M, 8):(1, 4096)`. A kernel takes its first 128 rows.

Composing with `128:1` walks the first mode. Its stride 1 splits that mode
exactly, so the walk enters it with all `M` rows. The rest, 128 rows, must then
either fit inside the `M` rows or be divisible by `M`.

When `M` is only positive, neither is proven: `M` could be 64. Composition
refuses with `ShapeDivisibility` and `Unproven`. When `M` is a multiple of 128,
`M >= 128` is proven, the tile fits, and the result is `128:1`. The symbolic
tests run both cases.

## Soundness beyond PyCuTe

PyCuTe composes a hierarchical right layout one mode at a time. Mode by mode,
`(4, 3):(0, 1)` composed with `(2, 3):(2, 1)` is `(2, 3):(0, 0)`. The right
layout's last element is `B(1, 2) = 2 + 2 = 4`, and `A(4)` is 1, since index 4
is coordinate `(0, 1)` of the left layout. The composed layout gives 0.

Composition means `A(B_0(c_0) + B_1(c_1))`, and mode by mode computes
`A(B_0(c_0)) + A(B_1(c_1))`. The two agree only when the sum never carries
across a mode of `A`. Cecka's Equation 23 states two sufficient conditions:
every summed walk splits the modes of `A` exactly, and every pair of walks on
one axis has segregated images, `s_i * d_i <= d_j` or `s_j * d_j <= d_i`.

tiki-cute checks both. `summed_axes` in [compose.rs](src/algebra/compose.rs)
finds the codomain axes where several walks add, checks their images, and marks
those walks `Walk::Summed`. A summed walk must split its mode exactly. A sole
walk keeps PyCuTe's allowance to stop short of the end of a mode its stride
does not divide, which is correct when no other walk adds to its offsets, as in
`(5, 3):(7, 1)` composed with `2:3`.

A divide is a composition with a tile and its complement, which are segregated
by construction. Dividing a flat matrix by a tile touches single-mode axes of
`A`, where no condition applies. The stricter rule refuses only compositions
that cut across a hierarchical mode in a way that carries. The random
postcondition tests found both failures. Every case in PyCuTe's own test suite
still matches.

XOR strides add a second kind of carry, a carry inside one mode. `16:^1`
composed with `4:3` walks the indices 0, 3, 6 and 9 of `16:^1`, whose offsets
are `^0`, `^3`, `^6` and `^9`. PyCuTe multiplies the XOR stride by the walk's
stride and returns `4:^3`, which maps 3 to `3 * ^3 = ^5`, since `3 * 3` carries
where the carry-less product does not. PyCuTe's sums of walks, layout sums and
inverses break the same way, and tiki-cute's checks at every index refuse each
of them. `tests/xor.rs` proves every case: for each it takes PyCuTe's
result, finds the index where it breaks the operation's contract, and requires
tiki-cute's refusal or smaller result:

| PyCuTe returns | Broken at | tiki-cute |
| --- | --- | --- |
| `16:^1 ∘ 4:3 = 4:^3` | index 3: `^5`, not `^9` | refuses composition |
| `16:^1 ∘ (3, 2):(1, 3) = (3, 2):(^1, ^3)` | index 4: `^2`, not `^4` | refuses composition |
| `12:^1 + (3, 4):(^16, ^32) = (3, 4):(^17, ^35)` | index 5: `^1`, not `^5` | refuses addition |
| `right_inverse((3, 8):(^1, ^5)) = 3:^1` | index 2: `A(^2) = ^4` | returns `1:0` |
| `left_inverse((3, 2):(^1, 0)) = 3:1` | index 2: `A(L(^2)) = ^1` | refuses |
| `logical_product(4:1, 2:^4) = (4, 2):(1, ^16)` | strides 1 and `^16` have no sum | refuses |

PyCuTe also keeps the XOR zero `F0` apart from the integer 0 and raises for
`16:F0 + 16:1`. tiki-cute reads `^0` as 0, which adds to every codomain, and
returns `16:1`. None of these calls is in PyCuTe's test suite, and every
recorded `F2` case matches.

## XOR strides

PyCuTe calls XOR strides `F2`, and tiki-cute follows its `F2` paths operation
by operation:

| Operation | With XOR strides |
| --- | --- |
| evaluation | each coordinate multiplies its stride carry-lessly, and the products add by XOR |
| `Shape::idx2crd_xor`, `Shape::crd2idx_xor` | an XOR index splits by carry-less division, and a shape whose extents carry into their prefix products is refused |
| coalesce | an XOR mode merges only across a power-of-two extent |
| composition | an integer walk multiplies A's XOR steps carry-lessly, and an XOR walk divides carry-lessly and must end in one mode of A |
| complement | refused, since a gap would be an XOR quotient |
| right inverse | PyCuTe's chain with its residue correction |
| left inverse | only a single mode of stride `^1` inverts |
| layout addition | the steps add by XOR |
| recast | only the stride `^1` recasts |
| products and divides | through complement and composition |

### Static extents

A carry-less product needs every bit of the integer it multiplies, and a launch
parameter has no bits the compiler knows. A layout with XOR strides therefore
has static extents. `Layout::new` refuses XOR strides beside a dynamic extent,
`Layout::call` refuses a negative or dynamic coordinate, and an operation that
would place a dynamic extent opposite an XOR stride refuses with
`LayoutError::XorOperand`. The swizzles CUTLASS applies to shared memory are
static too.

### Results checked at every index

The algebra computes with integers. Composition assumes `A(k * d) = k * A(d)`
for each walk and `A(x + y) = A(x) + A(y)` for each sum of walks. With XOR
strides the integer product or sum on the left must equal its carry-less
counterpart on the right, which holds exactly where no bit carries. The
condition depends on every coordinate a walk reaches. Coalescing has an exact
structural rule, a power-of-two extent, and uses it. For the other operations,
tiki-cute evaluates the result at every index, which is exact because an XOR
layout is static, and refuses with `Condition::CarryFree` at the first index
that differs:

- A composition is compared with `A(B(i))` at every index `i` whose `B(i)` is a
  coordinate of `A`, where the contract of composition holds.
- A layout sum is compared with the pointwise sum.
- A left inverse is checked for `A(L(A(i))) = A(i)`.
- A right inverse is cut instead of refused. Every prefix of its chain of modes
  is a smaller right inverse, so the longest prefix whose modes hold at every
  index is kept. Its indices are XOR indices, which `A` splits carry-lessly.
  That split agrees with the integer split on power-of-two extents, where
  PyCuTe's whole chain survives.

The check costs one evaluation per index of the result. For a swizzled
shared-memory tile that is a few thousand evaluations.

## Tensors and bounds

A [`Tensor`](src/tensor.rs) is a static layout placed at an element offset in
storage whose byte range is known. `Tensor::new` computes the least and
greatest offset of the layout, one mode at a time, so negative strides and
broadcast modes are exact. It refuses any tensor whose bytes could leave the
range. The fields are private, and slicing, composition and recasting pass
their result through `Tensor::new` again.

The range a tensor proves is its allocation. A tile that runs past the logical
end of a tensor may read bytes of the same allocation that belong to no
element, and the tensor accepts it. A
kernel often wants exactly that. Reading a whole tile without predication and
discarding the lanes past the logical end is faster than masking every load.
Memory safety needs only the bounds proof. Keeping the discarded lanes out of
the results is a separate proof about the flow of values, which the kernel
compiler owns.

## Printing and parsing

Layouts print as the `tiki.layout` Python API prints them, so one layout reads the
same in Python, in Rust and in a compiler error:

- `Display`: `Layout(shape=(4, 4), stride=(4, 1))`.
- `Layout::cute`: CuTe's `(4, 4):(4, 1)`.
- `Layout::describe`: one row per coordinate with its extent and stride, then
  the index formula.

[parse.rs](src/parse.rs) reads CuTe notation, including basis strides such as
`1@0 + 2@1`, XOR strides such as `^9`, and parameter names. The reference cases
and the tests are written in it. PyCuTe prints an XOR stride as `F9`, which
would read as a parameter name, so tiki-cute prints and parses `^9`.

## Verification

Each claim above has a test that exercises it:

| Test | Establishes |
| --- | --- |
| `tests/reference.rs` | every integer and `F2` call in PyCuTe's own test suite at `111253d`, 2991 of them, 427 with `F2` strides, returns PyCuTe's layout or refuses where PyCuTe raises |
| `tests/postconditions.rs` | each operation's contract on random layouts, checked at every index |
| `tests/symbolic.rs` | PyCuTe's symbolic cases, facts that prove a tile fits, and every symbolic result satisfying the concrete contracts after substitution |
| `tests/tensor.rs` | a tensor accepts exactly the placements whose bytes lie in bounds, and derived tensors stay in their storage |
| `tests/xor.rs` | each operation's contract on random XOR layouts, a `Swizzle` and its XOR layout agreeing at every index, the carry-free split of an XOR index, and each difference from PyCuTe's `F2` paths |

[record.py](tests/reference/record.py) records the reference cases by running
PyCuTe's test suite with the public operations wrapped. It keeps the calls
whose arguments are integer or `F2` layouts and tilers, and writes `F2`
strides in tiki-cute's notation. A caller that tiles nothing
calls no operation, so the replay skips PyCuTe's whole-layout `None` tiler by
name.

## Scope and open decisions

- **XOR components.** PyCuTe allows an `F2` value as one component of an
  arithmetic tuple, so one axis of a coordinate layout can be swizzled.
  tiki-cute's arithmetic tuples have integer components, and a tiler with XOR
  strides cannot move onto an axis.
- **XOR complements.** A complement of an XOR layout exists in F2, but its gaps
  are XOR quotients, which PyCuTe refuses and tiki-cute refuses too. Products
  of XOR layouts wait on it.
- **Maximal XOR right inverses.** A right inverse is cut to whole modes. A
  mode could be cut to a prefix of its extent instead, so `(3, 8):(^1, ^5)`
  has the right inverse `2:^1`, where tiki-cute returns `1:0`.
- **Composed layouts.** The `tiki.layout` API composes a swizzle, an offset and a
  layout into one `ComposedLayout`. Its Rust counterpart belongs here, beside
  `Swizzle`.
- **Overflow.** `Int` arithmetic panics on `i64` overflow, which only a layout
  spanning more than 2^63 elements reaches. The algebra runs in the compiler
  process, where a panic ends one compilation. `Tensor` uses checked arithmetic
  throughout and reports overflow as an error, because host code builds tensors
  from user shapes.
- **Upper bounds.** A parameter's facts give it a least value and a divisor.
  A fact such as `M <= 4096` would prove more, and `Param` is where it would
  live.
