.. _tiki-runtime:

Runtime
=======

The runtime runs compiled kernels on a GPU. It owns every device address,
every stream, and the lifetime of every buffer while the device uses it. It
is Rust, and only one crate in it calls the driver.

.. list-table::
   :header-rows: 1

   * - Crate
     - Owns
   * - ``tiki-cuda-sys``
     - The only calls into ``libcuda``: devices, streams, events, and device
       memory as owned values.
   * - ``tiki-cuda-runtime``
     - The allocator's cache and limits, leases, and batch completion.
   * - ``tiki-core``
     - The graph, its evaluator, and the order of streams.

Compiling and running
---------------------

Compiling a kernel and running it are separate steps, and only running
touches the driver.

- **Compiling** is ordinary Rust. The compiler traces a ``tk`` kernel, proves
  it with ``tiki-cute``, and lowers it to LLVM IR. LLVM's NVPTX backend writes
  PTX. No driver call and no ``unsafe`` take part.
  :ref:`tiki-kernels` follows one kernel through each stage.
- **Running** goes through ``libcuda``, the user-space interface to NVIDIA's
  kernel driver. Loading PTX, allocating memory, copying, launching, and
  waiting are driver calls, and every call into C is ``unsafe`` in Rust.
  ``tiki-cuda-sys`` confines those calls, as the C library confines system
  calls under Rust's standard library.

A kernel is safe when its proofs hold and the lowering keeps them. Host code
is safe when the values ``tiki-cuda-sys`` hands out admit no misuse.

Driver values
-------------

``tiki-cuda-sys`` hands out ``Device``, ``Stream``, ``Event``, and
``DeviceMemory``. Each owns what it names and releases it on drop. No address
or driver handle leaves the crate as an integer, so safe code cannot name
memory it does not own.

.. code-block:: rust

   let stream = Arc::new(Stream::new(Device::open(0)?)?);
   let mut memory = DeviceMemory::allocate(&stream, 64)?;
   stream.write(&[1.0f32; 16], &mut memory)?;
   let mut host = [0.0f32; 16];
   stream.read(&memory, &mut host)?;

- Every copy checks its length against the memory it reads or writes.
- Copies to and from host memory finish before they return, so the driver
  never uses a borrowed host slice after its borrow ends.
- Host slices hold ``Element`` types, plain numbers with no invalid bit
  patterns, so any bytes the device writes are valid values.
- Each call makes its device's context current and restores the caller's
  context afterwards, also when the call fails or panics.

Three failures stop the process instead of returning: a context that cannot
be restored, a host copy whose wait fails, and a batch handler that panics.
Each would otherwise leave a thread on an unknown device, a copy running into
a slice that has been returned, or retirement waiters blocked forever.

Memory lifetime and stream order
--------------------------------

Three questions decide when device memory may change hands, and each has its
own mechanism.

.. list-table::
   :header-rows: 1

   * - Question
     - Mechanism
   * - Does a program still hold the array?
     - Python reference counts and the graph's references.
   * - Does queued device work still use the memory?
     - A lease in the batch of that work.
   * - Has a producer on another stream finished before a consumer reads?
     - An event recorded after the producer.

Leases
~~~~~~

A launch or copy leases the storage of each argument to the current batch of
its stream. The completion worker drops the batch after the device passes it,
so the last reference to storage goes only after its last use, and the storage
then returns to the allocator's cache. Freeing and reuse wait on no stream.

The C++ CUDA backend follows the same rule by convention: ``eval.cpp`` adds
each input to the command encoder's temporaries, and the commit releases them
in a completion handler. The generated launcher takes the lease itself, so no
call site can leave it out.

Donation
~~~~~~~~

Leases hold the memory, not the array, so the graph still decides buffer
donation from its own reference counts, as ``array::is_donatable`` does.
Storage adds one condition: every read it still has in flight is on the
donating operation's stream. Stream order then finishes those reads before the
donated write. A read in flight on another stream blocks the donation, since
nothing orders it before the write, and the operation writes fresh storage
instead.

Stream order
~~~~~~~~~~~~

Work on one stream runs in order and needs nothing more. Across streams, the
graph decides, before any work is queued.

The evaluator walks the tape and marks each array that an operation on another
stream consumes. Right after it queues a marked array's producer, the array's
storage records its event on that stream, and the consumer's stream waits on
that event before it reads. The wait covers exactly the producer, and only
operations with a consumer on another stream record an event. Across
evaluations, an array's own event does the same. This is the evaluator of
``tiki/transforms.cpp``, where ``needs_fence`` holds the marks and each
stream's fence is updated after a marked producer; ``tiki-core`` carries it
over with the graph.

A stream can still reach storage the evaluator did not mark, as when a runtime
caller outside the graph copies between streams. The storage then records its
event at hand-off, on the stream of its last write, and the new stream waits
on it. That wait covers the write and anything queued after it on the same
stream, so it can wait longer than needed, never shorter.

Both cases follow one rule: a stream waits on the event of the storage's last
write before it reads, and also on the events of its reads before it writes.
A mark only moves the record earlier.

Cost
~~~~

Host time per queued operation on a GH200 with CUDA 13.3, over 20,000
operations:

.. list-table::
   :header-rows: 1

   * - Ordering
     - Fill
     - Copy
   * - None
     - 0.76 µs
     - 1.88 µs
   * - An event created and recorded after every operation
     - 1.50 µs
     - 3.82 µs
   * - A hand-off event only when the stream changes
     - 0.77 µs
     - 1.72 µs

A fill that changes stream every time costs 1.36 µs with hand-offs. Recording
a reused event costs 0.19 µs, and a stream wait 0.19 µs. An event after every
operation gives exact waits and charges every operation. A hand-off leaves
work on one stream free and can wait for work queued after the write. Marks
from the evaluator give exact waits and charge only operations with a consumer
on another stream.

Batch completion
----------------

A committed batch runs on the completion worker once the device finishes the
work queued before its commit. The stream's own signal stream waits for the
compute stream and launches a host function that marks the batch done; the
worker then runs the batch's handlers and drops what it leased. Each compute
stream has its own signal stream, so a slow stream never delays another
stream's batches. A batch whose callback cannot be scheduled stays leased for
the life of the process, since the device may still use what it holds.
