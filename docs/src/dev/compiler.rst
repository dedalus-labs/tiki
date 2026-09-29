.. _tiki-compiler:

Compiler
========

The compiler turns a ``tk`` kernel into PTX. Tiki writes the stages that know
what a kernel means: the kernel IR, the proofs, and the per-thread arithmetic.
LLVM and clang do the rest. Decades of optimizer and code generator work live
there, and Tiki's job is to hand them IR they can trust.

.. list-table::
   :header-rows: 1

   * - Stage
     - Form
     - Checked by
   * - Kernel IR
     - Tensors with ``tiki-cute`` layouts, tiles, and thread-value partitions,
       as the tracer or the graph compiler produces them.
     - The proofs.
   * - Proofs
     - Bounds, disjoint writes, coverage, barrier coverage and reach, each
       decided from layouts and the launch facts.
     - A kernel that fails one does not compile.
   * - Thread IR
     - One thread's program in SSA form. Layout evaluation has become integer
       arithmetic, and every integer carries its range over every admitted
       launch.
     - An interpreter checked against ``tiki-cute`` at every coordinate.
   * - LLVM IR
     - A typed model of LLVM IR in Rust. Its printer is the only code that
       writes LLVM syntax.
     - LLVM's verifier, and golden tests beside clang's IR for the same
       kernel written in CUDA C++.
   * - PTX
     - clang of the pinned LLVM release, run as a separate process.
     - ``ptxas`` as a check, and ``libNVVM`` compiling the same IR as a
       differential check.
   * - GPU binary
     - The driver compiles the PTX when the runtime loads it.
     - The runtime's tests on hardware.

LLVM and clang
--------------

Tiki uses LLVM's optimizer and its NVPTX backend, and clang as both the driver
and the reference frontend. One LLVM release, 21.1.8, supplies clang, ``opt``,
and ``llc``, and the compiler refuses any other. clang and LLVM are released as
one version ([Tips]_), and LLVM's textual IR carries no compatibility promise
across releases ([Policy]_), so a second release would be a second compiler.

The compiler emits LLVM IR as text. The in-memory, bitcode, and textual forms
of LLVM IR are equivalent, and parsing text runs LLVM's verifier ([LangRef]_).
Text keeps LLVM out of Tiki's process: an LLVM crash ends one compilation with
a typed error, and Tiki's own code needs no ``unsafe``. The text comes from a
typed model, so a width mismatch or a flag without its proof is a Rust type
error before LLVM sees the module.

PTX comes from one clang invocation:

.. code-block:: text

   clang --target=nvptx64-nvidia-cuda -march=sm_90 -O3 -S kernel.ll -o kernel.ptx

Kernels use NVPTX's conventions ([NVPTX]_): the ``ptx_kernel`` calling
convention, global memory in ``addrspace(1)``, special registers through
``llvm.nvvm.read.ptx.sreg.*``, and the block size through
``"nvvm.reqntid"``. Each module carries its data layout and triple, without
which LLVM enables no target-specific optimization ([Tips]_).

What Tiki promises LLVM
-----------------------

Every flag and attribute Tiki attaches is a promise. A violated ``nsw`` makes
its result poison, and poison taints everything computed from it ([UB]_). An
``inbounds`` GEP outside its allocation is poison ([GEP]_). An ``align`` means
that alignment or undefined behavior ([Tips]_). LLVM turns these promises into
faster code, and a false promise into wrong code.

clang makes these promises because C++ declares overflow and out-of-bounds
pointers undefined. Tiki makes them because its proofs establish them, and the
proofs establish more than a C++ frontend can know about the same kernel:

.. list-table::
   :header-rows: 1

   * - Proven fact
     - What Tiki emits
   * - An integer's range fits the width without wrapping
     - ``nsw``; ``nuw`` too when nothing involved is negative
   * - An index is never negative
     - ``zext nneg`` when widening, never ``sext``
   * - An offset lies inside its tensor
     - ``getelementptr inbounds nuw``
   * - A tensor is only read, and no argument aliases a written tensor
     - ``noalias`` and ``readonly`` on its parameter
   * - A parameter is a concrete value from the launcher
     - ``noundef``
   * - A launch fact bounds an integer parameter, such as ``512 <= N <= 2^48``
     - ``range(i64 512, 281474976710657)`` on the parameter
   * - The block has exactly 128 threads
     - ``"nvvm.reqntid"="128"``, and ``range(i32 0, 128)`` on the thread index
   * - The allocator aligns storage to 256 bytes and a thread's elements are
       consecutive and aligned
     - ``align 16`` on a ``<4 x float>`` access

The builder cannot emit a promise without the value that proves it. A flag in
the model is a field that holds its proof, so an unproven flag does not
compile.

Floating point stays IEEE unless the ``tk`` expression allows otherwise.
Fusing a multiply and an add into ``fma`` is the ``contract`` fast-math flag on
the instructions themselves: clang fuses an IR multiply and add that carry it
even under ``-ffp-contract=off``. The compiler writes ``contract`` only where
the kernel permits it.

Lowering
--------

The proofs hold for the kernel IR, and lowering keeps them true in LLVM IR:

- **Index width.** The proofs compute offsets as exact integers. Lowering uses
  32-bit arithmetic only where a proof bounds every value below 2\ :sup:`31`,
  and 64-bit arithmetic elsewhere, so tile arithmetic stays 32-bit and widens
  once for the global offset.
- **Floor division.** The layout algebra divides with floor division. LLVM's
  ``sdiv`` and ``srem`` round toward zero, so lowering emits ``udiv`` and
  ``urem`` for a dividend proven nonnegative, and an explicit floor sequence
  otherwise.
- **Launch facts.** A kernel runs only on arguments that satisfy the facts its
  proofs assumed, including a kernel that shapeless compilation reuses across
  shapes. The generated launcher checks each fact.
- **Shape.** The IR stays close to what clang emits for the same kernel in
  CUDA C++: SSA without ``alloca``, plain ``load`` and ``store`` on
  ``addrspace(1)``, and address arithmetic through GEPs. The further IR strays
  from clang's, the less LLVM optimizes it ([Tips]_).

An unproven ``nsw`` shows why promises need proofs. An offset
``row * 65536 + col`` lowered as 32-bit arithmetic with ``nsw`` and guarded by
``0 <= offset < len`` loses its ``offset >= 0`` check under ``opt -O2``, in
LLVM 21 and 23. At ``row = 32768`` the product wraps to -2\ :sup:`31`, passes
``offset < len``, and becomes +2\ :sup:`31` after ``zext nneg``: the store
lands 8 GiB past a 16-element buffer. The same IR without ``nsw`` keeps both
checks. The lowering tests keep this case.

Reference output
----------------

The same ``axpy`` (512-element tiles, 128 threads, four elements each) from
three compilers on a GH200 shows where LLVM expects IR to go:

.. list-table::
   :header-rows: 1

   * -
     - clang 21.1.8
     - nvcc 13.3
     - Triton 3.8, 16-byte alignment hints
   * - Index arithmetic
     - 32-bit in the tile, ``zext nneg`` to 64-bit
     - The same
     - 32-bit throughout
   * - Flags
     - ``nuw nsw``, ``or disjoint``, ``inbounds nuw``
     - Not visible
     - ``nuw nsw`` on thread arithmetic only; no ``inbounds``
   * - Loads and stores
     - LLVM ``load``; the read-only input becomes ``ld.global.nc``
     - ``ld.global.nc``
     - Inline PTX, ``ld.global.v4``
   * - Multiply-add
     - ``contract``, so ``fma``
     - ``fma``
     - ``fma``

clang's form is the model. Tiki keeps its shape, makes its promises from
proofs, and reaches Triton's 16-byte accesses from the layout and the
allocator's alignment rather than from launch-time hints.

Checks
------

- LLVM's verifier runs on every module.
- Golden tests compare the printed IR with clang 21.1.8's IR for the
  equivalent CUDA C++.
- The lowering interpreter agrees with ``tiki-cute`` at every coordinate of
  small shapes, at negative coordinates, and at offsets near 2\ :sup:`31` and
  2\ :sup:`32`.
- Optimized IR runs through ``-O2`` a second time. A change there means the
  pass order leaves work undone ([Tips]_).
- ``ptxas`` accepts the PTX, and ``libNVVM`` compiles the same IR as a
  differential oracle.

Alternatives
------------

- **NVIDIA's CuTe DSL** is a revocable EULA that forbids reverse engineering
  its compiler, so Tiki could not fix or ship it. The layout algebra and atoms
  are BSD-3 in CUTLASS, and Tiki implements them from there. CuTe DSL stays
  as a correctness and performance oracle.
- **CUDA Tile IR** has no thread index, warp, or shared memory operations, so
  it cannot express the thread-level control Tiki's kernels need.
- **cuda-oxide** compiles Rust kernels, but its shared memory, warp
  operations, and TMA require ``unsafe``.
- **libNVVM** runs the same device optimizer as ``nvcc`` but is closed. LLVM's
  NVPTX backend accepts the same IR, and Tiki's atoms are inline PTX, so new
  instructions do not wait for NVPTX intrinsics.
- **LLVM in process** through its C API would make every GEP an ``unsafe``
  call in Tiki's compiler and put LLVM's failures inside Tiki's process.

References
----------

.. [LangRef] `LLVM Language Reference Manual, Introduction
   <https://releases.llvm.org/21.1.0/docs/LangRef.html#introduction>`_.
.. [Policy] `LLVM Developer Policy, IR Backwards Compatibility
   <https://llvm.org/docs/DeveloperPolicy.html#ir-backwards-compatibility>`_.
.. [Tips] `Performance Tips for Frontend Authors
   <https://releases.llvm.org/21.1.0/docs/Frontend/PerformanceTips.html>`_.
.. [NVPTX] `User Guide for NVPTX Back-end, Conventions
   <https://releases.llvm.org/21.1.0/docs/NVPTXUsage.html#conventions>`_.
.. [UB] `LLVM IR Undefined Behavior Manual, Poison Values
   <https://releases.llvm.org/21.1.0/docs/UndefinedBehavior.html#poison-values>`_.
.. [GEP] `The Often Misunderstood GEP Instruction
   <https://releases.llvm.org/21.1.0/docs/GetElementPtr.html#what-happens-if-an-array-index-is-out-of-bounds>`_.
