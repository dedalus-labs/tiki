.. _unified_memory:

Unified Memory
==============

.. currentmodule:: tiki

Apple silicon has a unified memory architecture. The CPU and GPU have direct
access to the same memory pool. Tiki is designed to take advantage of that.

Concretely, when you make an array in Tiki you don't have to specify its location:


.. code-block:: python

  a = tk.random.normal((100,))
  b = tk.random.normal((100,))

Both ``a`` and ``b`` live in unified memory.

In Tiki, rather than moving arrays to devices, you specify the device when you
run the operation. Any device can perform any operation on ``a`` and ``b``
without needing to move them from one memory location to another. For example:

.. code-block:: python

  tk.add(a, b, stream=tk.cpu)
  tk.add(a, b, stream=tk.gpu)

In the above, both the CPU and the GPU will perform the same add
operation. The operations can (and likely will) be run in parallel since
there are no dependencies between them. See :ref:`using_streams` for more
information the semantics of streams in Tiki.

In the above ``add`` example, there are no dependencies between operations, so
there is no possibility for race conditions. If there are dependencies, the
Tiki scheduler will automatically manage them. For example:

.. code-block:: python

  c = tk.add(a, b, stream=tk.cpu)
  d = tk.add(a, c, stream=tk.gpu)

In the above case, the second ``add`` runs on the GPU but it depends on the
output of the first ``add`` which is running on the CPU. Tiki will
automatically insert a dependency between the two streams so that the second
``add`` only starts executing after the first is complete and ``c`` is
available.

A Simple Example
~~~~~~~~~~~~~~~~

Here is a more interesting (albeit slightly contrived example) of how unified
memory can be helpful. Suppose we have the following computation:

.. code-block:: python

  def fun(a, b, d1, d2):
    x = tk.matmul(a, b, stream=d1)
    for _ in range(500):
        b = tk.exp(b, stream=d2)
    return x, b

which we want to run with the following arguments:

.. code-block:: python

  a = tk.random.uniform(shape=(4096, 512))
  b = tk.random.uniform(shape=(512, 4))

The first ``matmul`` operation is a good fit for the GPU since it's more
compute dense. The second sequence of operations are a better fit for the CPU,
since they are very small and would probably be overhead bound on the GPU.

If we time the computation fully on the GPU, we get 2.8 milliseconds. But if we
run the computation with ``d1=tk.gpu`` and ``d2=tk.cpu``, then the time is only
about 1.4 milliseconds, about twice as fast. These times were measured on an M1
Max.
