.. _tiki-kernels:

Kernels from building blocks
============================

A Tiki kernel is written from seven primitives and a few algebraic
structures. This page names the structures, then builds each common kernel
from them. The code shows the kernel language Tiki's compiler implements.

Algebraic structures
--------------------

Each structure is a set with operations and the laws they obey. Each one adds
a law to the one before, and each law is what some part of a kernel relies on.

.. list-table::
   :header-rows: 1

   * - Structure
     - Adds
     - Example
   * - magma
     - a binary operation ``⊕``
     - ``max`` on floats
   * - semigroup
     - associativity: ``(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)``
     - ``max``
   * - monoid
     - an identity ``e``: ``e ⊕ a = a ⊕ e = a``
     - ``(max, −∞)``, ``(+, 0)``
   * - group
     - an inverse for every element: ``a ⊕ a⁻¹ = e``
     - integers under ``+``
   * - semiring
     - a second monoid ``⊗`` that distributes over a commutative ``⊕``, with
       ``e⊕ ⊗ a = e⊕``
     - ``(max, +)``
   * - ring
     - a semiring whose ``⊕`` is a group
     - the integers
   * - field
     - a commutative ring whose nonzero elements have ``⊗``-inverses
     - the reals, and F2
   * - module
     - an abelian group ``M`` scaled by a ring ``R``, with ``r ⊗ (a ⊕ b) = r ⊗ a ⊕ r ⊗ b``
     - integer strides, arithmetic-tuple strides
   * - vector space
     - a module over a field
     - XOR strides

**F2** is the field of two elements, ``{0, 1}``. Its addition is XOR and its
multiplication is AND, and every element is its own additive inverse. Read an
offset as a vector of bits over F2, and XOR strides make a layout a linear map
over F2. A swizzle is such a layout.

A layout's strides always form a module over its coordinates' ring.
``L(c) = Σ cᵢ ⊗ dᵢ`` is the inner product of the coordinate with the strides,
and only the module decides what ``⊕`` and ``⊗`` mean. The integers give
ordinary strides, integer tuples give coordinate layouts, and F2 gives XOR
strides.

Floating-point ``+`` is associative only up to rounding. Regrouping a fold of
floats changes its rounding and nothing else, and kernels regroup freely for
that reason.

Operations
----------

.. list-table::
   :header-rows: 1

   * - Operation
     - Definition
     - In ``tk``
   * - map
     - Apply ``f`` to every element. Arrays with ``map`` form a functor.
     - ``f(x)`` on a tensor
   * - zip
     - Pair the elements of two arrays at each coordinate.
     - ``tk.zip(x, y)``
   * - fold
     - Combine every element with a monoid: ``e ⊕ x₀ ⊕ x₁ ⊕ … ⊕ xₙ₋₁``. Also
       called a reduction.
     - ``tk.fold(x, monoid)``
   * - scan
     - The fold of every prefix: ``yᵢ = x₀ ⊕ … ⊕ xᵢ``. Inclusive as written,
       exclusive when ``yᵢ`` stops before ``xᵢ``. Also called a prefix sum.
     - ``tk.scan(x, monoid)``
   * - segmented fold, scan
     - A fold or scan of the segmented monoid below.
     - ``tk.segmented(monoid)``
   * - tile
     - Divide a tensor by a tiler: mode 0 is a tile, mode 1 indexes the tiles.
     - ``x.tile(shape)``
   * - partition
     - Compose a tile with a thread-value layout and slice it at one thread's
       index.
     - ``x.partition(threads, index)``
   * - rank, select
     - For a set of positions: how many members precede a position, and which
       position is the ``k``-th member.
     - ``tk.Bitmap``

The common monoids and semirings of kernels:

.. list-table::
   :header-rows: 1

   * - Semiring
     - ``⊕``, identity
     - ``⊗``, identity
     - Used for
   * - real
     - ``+``, 0
     - ``×``, 1
     - matrix multiplication
   * - tropical
     - ``max``, −∞
     - ``+``, 0
     - shortest and longest paths, Viterbi
   * - log
     - ``logsumexp``, −∞
     - ``+``, 0
     - probabilities in log space
   * - boolean
     - ``or``, false
     - ``and``, true
     - reachability, masks

Matrix multiplication is defined over any semiring:
``C[i, j] = ⊕_k A[i, k] ⊗ B[k, j]``. Log-sum-exp is the addition of the log
semiring, ``a ⊕ b = log(eᵃ + eᵇ)``, so a softmax normalizer is the ``⊕``-fold
of the scores in that semiring.

**Online softmax** folds a monoid of triples ``(m, s, o)``: the running
maximum, the sum of ``exp(x - m)``, and the output weighted by those
exponentials. Two partial results combine by rescaling both to the larger
maximum:

.. code-block:: text

   m = max(m₁, m₂)
   (m₁, s₁, o₁) ⊕ (m₂, s₂, o₂) = (m, s₁·e^(m₁−m) + s₂·e^(m₂−m), o₁·e^(m₁−m) + o₂·e^(m₂−m))
   identity = (−∞, 0, 0)

``(m, s)`` is the log-semiring sum, kept as a maximum and a rescaled sum so
that no exponential overflows. Attention is this fold over the key tiles.

A **segmented** monoid lifts any monoid to pairs of a flag and a value, where
the flag marks the first element of a segment:

.. code-block:: text

   (f₁, a) ⊕ (f₂, b) = (f₁ or f₂, b if f₂ else a ⊕ b)
   identity = (false, e)

The lift is associative, so one scan handles every segment of a ragged batch
at once. Variable-length attention and mixture-of-experts routing are
segmented folds and scans.

.. code-block:: python

   import tiki as tk

   tk.sum           # Monoid(+, 0)
   tk.max           # Monoid(max, -inf)
   tk.logsumexp     # Monoid(logsumexp, -inf)
   tk.real          # Semiring(tk.sum, tk.product)
   tk.tropical      # Semiring(tk.max, tk.sum)
   tk.log           # Semiring(tk.logsumexp, tk.sum)
   tk.segmented(tk.sum)

Primitives
----------

.. list-table::
   :header-rows: 1

   * - Primitive
     - In ``tk``
     - What it does
   * - ``Layout``
     - ``tk.Layout(shape, stride)``
     - Maps coordinates to offsets, for data and for work.
   * - ``Tensor``
     - ``tk.shared(...)``, ``tk.registers(...)``
     - An accessor composed with a layout, in one memory space.
   * - ``Copy``
     - ``tk.copy(src, dst)``
     - Moves one tensor into another with the target's copy instruction.
   * - ``Mma``
     - ``tk.mma(a, b, acc)``
     - Adds ``a @ b`` to ``acc`` on the matrix unit, over the real semiring.
   * - ``Shuffle``
     - ``tk.shuffle(value, lane)``
     - Exchanges values between the lanes of one unit.
   * - ``Barrier``
     - ``tk.Barrier(...)``
     - Orders a producer before its consumers.
   * - ``Atomic``
     - ``tk.atomic(dst, value, monoid)``
     - Combines a value into memory other units also write.

A tensor's accessor is usually a ``Pointer`` into one memory space. The
coordinate accessor returns its offset unchanged and reads no memory, which is
how ``tk.copy`` masks the elements of a partial tile.

Tensors carry two layout operations that every kernel uses:

- ``x.tile(shape)`` divides ``x`` into tiles of ``shape``: mode 0 is a tile and
  mode 1 indexes the tiles.
- ``x.partition(threads, index)`` composes ``x`` with a thread-value layout and
  slices it at one thread's index, giving that thread's elements.

Elementwise
-----------

Partition, copy in, compute, copy out.

.. code-block:: python

   @tk.kernel(block=128)
   def axpy(alpha: float, x: tk.Tensor, y: tk.Tensor):
       # 128 threads each own 4 consecutive elements of a 512-element tile.
       threads = tk.Layout((128, 4), stride=(4, 1))
       xs = x.tile(512)[:, tk.block.index].partition(threads, tk.thread.index)
       ys = y.tile(512)[:, tk.block.index].partition(threads, tk.thread.index)

       xr = tk.copy(xs, tk.registers)
       yr = tk.copy(ys, tk.registers)
       tk.copy(alpha * xr + yr, ys)

Fold
----

A fold combines within each thread, then across a warp with ``Shuffle``, then
across warps through shared memory.

.. code-block:: python

   @tk.kernel(block=256)
   def row_sums(x: tk.Tensor, out: tk.Tensor):
       threads = tk.Layout((256, 4), stride=(4, 1))
       row = x[tk.block.index, :]

       total = tk.sum.identity
       for chunk in row.tile(1024).tiles():
           values = tk.copy(chunk.partition(threads, tk.thread.index), tk.registers)
           total = tk.sum(total, tk.fold(values, tk.sum))
       total = tk.fold(total, tk.sum, over=tk.block)

       if tk.thread.index == 0:
           out[tk.block.index] = total

Replacing ``tk.sum`` with ``tk.max`` gives a row maximum, and with
``tk.logsumexp`` a row log-normalizer. The kernel is the same.

Scan
----

A scan folds each prefix. Across blocks, each block publishes its total and
reads its predecessors' totals, a pattern called decoupled look-back. The
segmented monoid restarts the scan at each sequence of a packed batch.

.. code-block:: python

   @tk.kernel(block=256)
   def cumulative_sums(x: tk.Tensor, starts: tk.Tensor, out: tk.Tensor):
       # starts[i] is true where a new sequence begins.
       tk.scan(tk.zip(starts, x), tk.segmented(tk.sum), into=out)

Mixture-of-experts routing is a scan too. Counting the tokens routed to each
expert and scanning the counts gives each token its slot in a buffer grouped
by expert, which a grouped matrix multiply then reads in order.

Transpose
---------

A transpose copies through shared memory. Reading a column of a plain 32 by 32
tile touches one bank 32 times. XOR strides store row ``r``'s column ``c`` at
``32·r + (c xor r)``, so a column's 32 elements land in 32 different banks.

.. code-block:: python

   @tk.kernel(block=(32, 8))
   def transpose(x: tk.Tensor, y: tk.Tensor):
       threads = tk.Layout((32, 8, 4), stride=(1, 32, 256))
       src = x.tile((32, 32))[:, :, tk.block.y, tk.block.x]
       dst = y.tile((32, 32))[:, :, tk.block.x, tk.block.y]

       # Row r steps by 32 xor 1 = 33 in F2, column c by 1.
       staging = tk.shared(x.dtype, tk.Layout((32, 32), stride=(tk.xor(33), tk.xor(1))))

       tk.copy(src.partition(threads, tk.thread.index), staging.partition(threads, tk.thread.index))
       tk.Barrier(tk.block).sync()
       tk.copy(staging.T.partition(threads, tk.thread.index), dst.partition(threads, tk.thread.index))

Pipeline
--------

A pipeline is a ring of shared-memory stages. ``Copy`` fills a stage while the
consumer reads an earlier one, and each stage's barrier records when its copy
has landed. ``tk.Pipeline`` owns the ring and the barriers, and yields each
stage once its copy is complete.

.. code-block:: python

   with tk.Pipeline(stages=3) as pipe:
       for stage in pipe.stream(a_tiles, b_tiles):
           consume(stage)

Matrix multiply
---------------

A pipeline streams tiles of ``A`` and ``B`` along ``K``, and ``Mma`` accumulates
each pair into registers.

.. code-block:: python

   @tk.kernel(block=128)
   def matmul(a: tk.Tensor, b: tk.Tensor, c: tk.Tensor):
       a_tiles = a.tile((128, 32))[:, tk.block.y, :]   # this block's rows, every K step
       b_tiles = b.tile((32, 128))[:, :, tk.block.x]   # this block's columns, every K step

       acc = tk.registers(c.dtype, tk.mma.accumulator((128, 128)))
       with tk.Pipeline(stages=3) as pipe:
           for a_tile, b_tile in pipe.stream(a_tiles, b_tiles):
               tk.mma(a_tile, b_tile, acc)

       tk.copy(acc, c.tile((128, 128))[:, tk.block.y, tk.block.x])

A matrix multiply over another semiring has no matrix instruction, so it
folds instead: ``tk.matmul(a, b, semiring=tk.tropical)`` combines the products
with the semiring's ``⊕``.

Attention
---------

Attention is two matrix multiplies joined by the online-softmax fold. Each
block owns one tile of queries and folds over the key tiles of the same
sequence.

.. code-block:: python

   @tk.kernel(block=128)
   def attention(q: tk.Tensor, k: tk.Tensor, v: tk.Tensor, o: tk.Tensor):
       queries = tk.copy(q.tile((128, q.shape[-1]))[:, tk.block.index], tk.shared)

       state = tk.online_softmax.identity
       with tk.Pipeline(stages=2) as pipe:
           for keys, values in pipe.stream(k.tile((64, k.shape[-1])), v.tile((64, v.shape[-1]))):
               scores = tk.mma(queries, keys.T)
               m = tk.fold(scores, tk.max, over=tk.rows)
               p = tk.exp(scores - m)
               state = tk.online_softmax(state, (m, tk.fold(p, tk.sum, over=tk.rows), tk.mma(p, values)))

       m, s, out = state
       tk.copy(out / s, o.tile((128, o.shape[-1]))[:, tk.block.index])

Variable-length attention packs every sequence into one tensor. The key loop
runs over the tiles of the query's own sequence, and the schedule below maps
blocks to the tiles that exist.

Scheduling ragged work
----------------------

A packed batch has a different number of tiles per sequence. A bitmap over
all possible tiles marks the ones that exist, and a running popcount turns it
into two constant-time maps, which ``tk.Bitmap`` provides:

- **rank**: the number of live tiles before a given tile, its slot in the work
  list;
- **select**: the ``k``-th live tile, the tile work item ``k`` processes.

A persistent kernel launches one block per core and has each block process
work items ``k, k + blocks, k + 2·blocks, …``, using select to find each
item's tile. Block-sparse attention masks and expert routing are the same
structure. On the host, large sparse sets of this kind are stored as Roaring
bitmaps, which keep dense regions as plain bitmaps and sparse regions as sorted
arrays.

Proofs
------

The compiler proves each kernel from the same structure it is built from:
bounds from each tensor's layout and its allocation, disjoint writes from
each partition, and completion order from each barrier and pipeline stage.
It also proves that every thread of a block reaches each block-wide barrier.
An index read from tensor data, as in a gather or a block table, has no static
proof, so the compiler checks it before the access.

From tk to PTX
--------------

Compiling a kernel is ordinary Rust and never calls the driver. Running it
goes through the driver, and only the launcher does that. This section
follows the elementwise ``axpy`` above through each stage.

**Layouts and proofs.** Tracing turns each tensor into a ``tiki-cute`` layout.
The launch fact ``512 | N`` is a parameter fact, and the proofs use it:

.. code-block:: rust

   use tiki_cute::{Int, Layout, Param, Shape, Tiler, Tuple};

   let n = Int::from(Param::multiple_of("N", 512)?);
   let array = Layout::from(Shape::try_from(Tuple::Leaf(n.clone()))?);   // N:1
   let tiles = array.logical_divide(&Tiler::try_from(512)?)?;             // (512, floor(N/512)):(1, 512)
   let threads: Layout = "(128, 4):(4, 1)".parse()?;
   let partition = tiles.compose(&Tiler::from([Some(Tiler::from(threads)), None]))?;
   // ((128, 4), floor(N/512)):((4, 1), 512): (thread, value, block) to element offset

   // Highest offset: 4·127 + 3 + 512·(floor(N/512) - 1) = N - 1, below N.
   prove::in_bounds(&partition, &array)?;
   // No two (thread, value, block) coordinates reach one offset of y.
   prove::injective(&partition)?;
   // N has no upper bound, so offsets lower as 64-bit integers.
   let width = IndexWidth::for_highest(&n);

With ``Param::positive("N")`` instead, the bounds question stays open for the
last tile. The compiler then guards it with ``offset < N`` or refuses the
kernel.

**Thread IR.** Layout evaluation becomes integer arithmetic, static loops
unroll, and each access keeps the fact that proves it:

.. code-block:: text

   axpy(alpha: f32, x: global f32[N], y: global f32[N], N: i64 {N > 0, N % 512 == 0})
     b: i64 = block.x                  range [0, N/512)
     t: i64 = thread.x                 range [0, 128)
     base: i64 = 512*b + 4*t           range [0, N-4], multiple of 4
     xr[v] = load x[base + v]          v in 0..4, base + 3 <= N - 1
     yr[v] = load y[base + v]
     store y[base + v] = alpha*xr[v] + yr[v]

**LLVM IR.** Every flag that lets LLVM assume something comes from a fact
above, or for ``noalias`` from the launcher's borrows:

.. code-block:: llvm

   define ptx_kernel void @axpy(float %alpha, ptr addrspace(1) noalias %x,
                                ptr addrspace(1) noalias %y, i64 %N) {
     %b    = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
     %t    = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
     %b64  = zext i32 %b to i64
     %t64  = zext i32 %t to i64
     %blk  = shl nuw nsw i64 %b64, 9                 ; 512*b < N < 2^63
     %lane = shl nuw nsw i64 %t64, 2
     %base = add nuw nsw i64 %blk, %lane
     %px   = getelementptr inbounds float, ptr addrspace(1) %x, i64 %base
     %py   = getelementptr inbounds float, ptr addrspace(1) %y, i64 %base
     %xv   = load <4 x float>, ptr addrspace(1) %px, align 16   ; base % 4 == 0
     %yv   = load <4 x float>, ptr addrspace(1) %py, align 16
     %av   = insertelement <4 x float> poison, float %alpha, i64 0
     %as   = shufflevector <4 x float> %av, <4 x float> poison, <4 x i32> zeroinitializer
     %r    = call <4 x float> @llvm.fma.v4f32(<4 x float> %as, <4 x float> %xv, <4 x float> %yv)
     store <4 x float> %r, ptr addrspace(1) %py, align 16
     ret void
   }

**PTX.** LLVM's NVPTX backend writes the PTX for ``sm_90``:

.. code-block:: text

   .visible .entry axpy(.param .f32 alpha, .param .u64 x, .param .u64 y, .param .u64 N)
   {
       mov.u32          %r1, %ctaid.x;
       mov.u32          %r2, %tid.x;
       mul.wide.u32     %rd1, %r1, 2048;
       mad.wide.u32     %rd2, %r2, 16, %rd1;
       ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd3];
       ld.global.v4.f32 {%f5, %f6, %f7, %f8}, [%rd4];
       fma.rn.f32       %f9, %f0, %f1, %f5;
       st.global.v4.f32 [%rd4], {%f9, %f10, %f11, %f12};
       ret;
   }

**Launch.** The compiler generates the launcher from the kernel's signature.
It is the only way to run the kernel, and it checks every fact the proofs
assumed:

.. code-block:: rust

   pub struct Axpy {
       function: Function,
   }

   impl Axpy {
       pub fn load(module: &Module) -> Result<Axpy, LoadError> {
           // The driver's parameter offsets and sizes must match the signature.
           let signature = [Parameter::F32, Parameter::Pointer, Parameter::Pointer, Parameter::I64];
           let function = module.function("axpy", &signature)?;
           Ok(Axpy { function })
       }

       pub fn launch(
           &self,
           stream: &Stream,
           alpha: f32,
           x: &DeviceArray<f32>,
           y: &mut DeviceArray<f32>,
           batch: &mut Batch,
       ) -> Result<(), LaunchError> {
           let n = x.len();
           if y.len() != n {
               return Err(LaunchError::Length { expected: n, found: y.len() });
           }
           if n == 0 || n % 512 != 0 {
               return Err(LaunchError::Fact { parameter: "N", fact: "a positive multiple of 512" });
           }
           let shape = LaunchShape { blocks: n / 512, threads: 128 };
           // Orders the launch after x's last write and y's last access, and
           // retains both arrays in the batch until the device passes it.
           stream.launch(&self.function, shape, (alpha, x, y, n as i64), batch)
       }
   }

``x`` is borrowed and ``y`` is borrowed mutably, so the borrow checker rejects
a call that passes one array as both. The only ``unsafe`` on this path is the
driver call inside ``Stream::launch``, and each of its preconditions is a
fact the types above already hold.
