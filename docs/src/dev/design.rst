.. _tiki-design:

Framework design
================

Tiki is built from a small set of nouns. Each one does a job no other noun
does, and together they cover a program from Python down to the processor.

.. list-table::
   :header-rows: 1

   * - Noun
     - What it is
   * - ``Array``
     - An immutable value with a shape and a dtype, possibly not yet computed.
   * - ``Op``
     - An operation on arrays, with one rule per functor.
   * - ``Function``
     - A composition of ops, traced from Python.
   * - ``Kernel``
     - A function lowered onto one processor.
   * - ``Tensor``
     - An accessor composed with a layout.
   * - ``Layout``
     - A map from coordinates to offsets.
   * - ``Accessor``
     - A map from offsets to values: a pointer into memory, or a function.
   * - ``Target``
     - A processor: its memory spaces and its execution units.
   * - ``Device``, ``Stream``
     - Where a kernel runs, and the order kernels run in.

Values and places
-----------------

An ``Array`` is a value. A ``Tensor`` is a place. An evaluated array is stored
in a tensor, but the two are different nouns for two reasons.

- An array that has not been computed has no layout yet. The compiler chooses
  the layout of every intermediate nobody has observed, which is what lets it
  tile, swizzle and fuse.
- Many tensors are not arrays: a shared-memory staging buffer, a register
  fragment, an accumulator in tensor memory. They are mutable, they exist
  inside one kernel, and no user composes them.

A tensor reads coordinate ``c`` as ``accessor(layout(c))``. The layout turns a
coordinate into an offset, and the accessor turns the offset into a value. An
accessor is usually a pointer into one memory space. It can also be a
function: the accessor that returns its offset unchanged gives a coordinate
tensor, which kernels use to mask partial tiles.

The word tensor follows CuTe. In mathematics and physics a tensor is a
multilinear object, and an array of numbers is its components in one basis.
Tiki's tensors are such component arrays, and a layout places them in memory.

A layout is not a basis. A change of basis mixes components: each new component
is a linear combination of the old ones. A change of layout moves components
without changing a value, so row-major and column-major store the same numbers
at different offsets. A bijective layout is a permutation of memory, the one
basis change that only reorders. A broadcast layout, with a zero stride, sends
many coordinates to one offset and is no basis at all.

Ops and their rules
-------------------

An ``Op`` has a forward computation and one optional rule per functor:

- **vjp**, the vector-Jacobian product, for reverse-mode derivatives;
- **jvp**, the Jacobian-vector product, for forward-mode derivatives;
- **vmap**, for batching.

The rules exist because the functors below are defined op by op. Once an op states
its rules, every function that uses it can be differentiated and batched,
however it composes. A researcher adds an op from Python:

.. code-block:: python

   import tiki as tk

   @tk.op
   def f(x):
       return tk.sin(x) * x

   @f.vjp
   def f_vjp(x, cotangent, output):
       return cotangent * (tk.cos(x) * x + tk.sin(x))

   @f.jvp
   def f_jvp(x, dx):
       return dx * (tk.cos(x) * x + tk.sin(x))

   @f.vmap
   def f_vmap(x, axis):
       return f(x), axis

Functors
--------

``grad``, ``jvp``, ``vjp`` and ``vmap`` take a function and return a function.
They trace and never run anything, so they have no side effects. Each one
preserves composition:

.. code-block:: text

   jvp(g ∘ f)  = jvp(g) ∘ jvp(f)     the chain rule
   vmap(g ∘ f) = vmap(g) ∘ vmap(f)
   vjp(g ∘ f)  = vjp(f) ∘ vjp(g)     reversed, which is why backprop runs backward

The ops generate a category of functions, and each of these maps is a functor
on it. A functor is fixed by what it does to the generators, which is why each
op carries one rule per functor.

``vmap`` turns a function of one example into a function of a batch, with no
loop. It is the same idea as a GPU kernel, whose body is written for one thread
and run for every thread. Functors compose in any order:

.. code-block:: python

   import tiki as tk

   def loss(w, x):
       return tk.sum((x @ w) ** 2)

   w = tk.ones((3,))
   xs = tk.random.normal((8, 4, 3))

   # One gradient per example in the batch, without a Python loop.
   per_example = tk.vmap(tk.grad(loss), in_axes=(None, 0))(w, xs)
   assert per_example.shape == (8, 3)

A kernel written thread by thread keeps control of its threads under ``vmap``.
Its vmap rule states where the batch goes. The default adds one mode to each
tensor's layout and one dimension to the launch, so each batch element gets its
own blocks.

Kernels
-------

A kernel is a function lowered onto one processor. Every function becomes
kernels by default. Tracing and the layout algebra are cheap next to code
generation, and code generation runs once per kernel: compiled kernels are
cached by their content, and dynamic extents avoid a compilation per shape.
Lazy evaluation without lowering remains available for debugging.

Kernel building blocks
----------------------

A kernel is written from seven primitives. Each one is a single hardware
capability, and each target supplies its instructions for it.

.. list-table::
   :header-rows: 1

   * - Primitive
     - What it does
     - Instructions
   * - ``Layout``
     - Maps coordinates to offsets, for data and for work.
     - integer arithmetic
   * - ``Tensor``
     - Names data in one memory space.
     - addressing
   * - ``Copy``
     - Moves one tensor into another.
     - vector load and store, ``cp.async``, TMA, ``ldmatrix``, DMA
   * - ``Mma``
     - Multiplies and accumulates tiles on the matrix unit.
     - ``mma.sync``, ``wgmma``, ``tcgen05.mma``, MFMA, MXU
   * - ``Shuffle``
     - Exchanges values between the lanes of one unit.
     - ``shfl.sync``, DPP, vector lane rotation
   * - ``Barrier``
     - Orders a producer before its consumers.
     - ``bar.sync``, ``mbarrier``, semaphores
   * - ``Atomic``
     - Combines a value into memory other units also write.
     - ``red``, ``atom``

Everything else is a composition of these, and the layout algebra does the
index arithmetic for each one:

- **Partition.** A thread's share of a tile is the tile composed with a
  thread-value layout, sliced at the thread's index.
- **Fold and scan.** A monoid is an associative combine with an identity. A
  fold combines each thread's values, then combines across lanes with
  ``Shuffle``, then across blocks with ``Atomic`` or a second pass. A scan does
  the same and carries each prefix forward.
- **Pipeline.** A ring of shared-memory tensors: ``Copy`` fills one stage while
  ``Mma`` reads another, and a ``Barrier`` per stage orders the two.
- **Elementwise.** Partition, copy in, compute, copy out.
- **Transpose.** A copy through a shared tensor with a swizzled layout.
- **Matrix multiply.** A pipeline feeding ``Mma`` on each thread's partition,
  then a copy out.
- **Attention.** Two matrix multiplies joined by the online softmax, which is a
  fold whose monoid carries the running maximum, sum and output.

The proofs follow the same structure: bounds from each tensor's layout,
disjoint writes from each partition, and completion order from each barrier.

Targets
-------

A ``Target`` names the two facts a lowering depends on: where storage can
live, and which units execute the work. The layout algebra and the kernel
language are the same on every target.

.. list-table::
   :header-rows: 1

   * - Target
     - Memory spaces
     - Execution units
   * - CUDA
     - global, shared, cluster shared, tensor memory, registers
     - thread, warp, warpgroup, block, cluster, grid
   * - AMD
     - global, local data share, registers
     - lane, wavefront, workgroup, grid
   * - TPU
     - HBM, vector memory, scalar memory, vector registers
     - lane, core, chip

A TPU core runs one program at a time. Its parallelism is in vector lanes and
the matrix unit, and data moves between memory spaces by explicit asynchronous
copies. The same layouts describe both models: a thread-value layout on a GPU
becomes a lane-value layout over vector registers. A researcher adds another
processor by implementing ``Target`` in the ``tiki-target`` crate. The
targets name the model each backend lowers to. CUDA is the first backend.
