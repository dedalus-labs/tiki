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
     - An operation on arrays, with the rules the transformations need.
   * - ``Function``
     - A composition of ops, traced from Python.
   * - ``Kernel``
     - A function lowered onto one processor.
   * - ``Tensor``
     - Storage seen through a layout.
   * - ``Layout``
     - A map from coordinates to offsets.
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

The word follows CuTe, where a tensor is an engine composed with a layout. In
mathematics and physics a tensor is a multilinear object whose components
change by a fixed rule when the basis changes. An array of numbers is its
components in one basis. Tiki's tensors are such component arrays, placed in
memory by a layout. They carry no change-of-basis rule.

Ops and their rules
-------------------

An ``Op`` has a forward computation and three optional rules:

- **backward**, the vector-Jacobian product, for reverse-mode derivatives;
- **tangent**, the Jacobian-vector product, for forward-mode derivatives;
- **batch**, for ``vmap``.

The rules exist because the transformations below are defined op by op. Once
an op states its rules, every function that uses it can be differentiated and
batched, however it composes. A researcher adds an op from Python:

.. code-block:: python

   import tiki as tk

   @tk.custom_function
   def f(x):
       return tk.sin(x) * x

   @f.vjp
   def f_backward(x, cotangent, output):
       return cotangent * (tk.cos(x) * x + tk.sin(x))

   @f.jvp
   def f_tangent(x, dx):
       return dx * (tk.cos(x) * x + tk.sin(x))

   @f.vmap
   def f_batch(x, axis):
       return f(x), axis

Transformations
---------------

``grad``, ``jvp``, ``vjp`` and ``vmap`` take a function and return a function.
They trace and never run anything, so they have no side effects. Each one
preserves composition, which is what makes it well defined:

.. code-block:: text

   jvp(g ∘ f)  = jvp(g) ∘ jvp(f)     the chain rule
   vmap(g ∘ f) = vmap(g) ∘ vmap(f)
   vjp(g ∘ f)  = vjp(f) ∘ vjp(g)     reversed, which is why backprop runs backward

In the language of category theory the ops generate a category of functions,
and each transformation is a functor on it. A functor is fixed by what it does
to the generators, which is why each op carries one rule per transformation.

``vmap`` turns a function of one example into a function of a batch, with no
loop. It is the same idea as a GPU kernel, whose body is written for one thread
and run for every thread. Transformations compose in any order:

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
Its batch rule states where the batch goes. The default adds one mode to each
tensor's layout and one dimension to the launch, so each batch element gets its
own blocks.

Kernels
-------

A kernel is a function lowered onto one processor. Every function becomes
kernels by default. Tracing and the layout algebra are cheap next to code
generation, and code generation runs once per kernel: compiled kernels are
cached by their content, and dynamic extents avoid a compilation per shape.
Lazy evaluation without lowering remains available for debugging.

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
