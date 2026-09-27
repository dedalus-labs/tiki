.. _numpy:

Conversion to NumPy and Other Frameworks
========================================

Tiki array supports conversion between other frameworks with either:

* The `Python Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_.
* `DLPack <https://dmlc.github.io/dlpack/latest/>`_.

Let's convert an array to NumPy and back.

.. code-block:: python

  import tiki as tk
  import numpy as np

  a = tk.arange(3)
  b = np.array(a) # copy of a
  c = tk.array(b) # copy of b

.. note::

    Since NumPy does not support ``bfloat16`` arrays, you will need to convert
    to ``float16`` or ``float32`` first: ``np.array(a.astype(tk.float32))``.
    Otherwise, you will receive an error like: ``Item size 2 for PEP 3118
    buffer format string does not match the dtype V item size 0.``

By default, NumPy copies data to a new array. This can be prevented by creating
an array view:

.. code-block:: python

  a = tk.arange(3)
  a_view = np.array(a, copy=False)
  print(a_view.flags.owndata) # False
  a_view[0] = 1
  print(a[0].item()) # 1

.. note::

    NumPy arrays with type ``float64`` will be default converted to Tiki arrays
    with type ``float32``.

A NumPy array view is a normal NumPy array, except that it does not own its
memory. This means writing to the view is reflected in the original array.

While this is quite powerful to prevent copying arrays, it should be noted that
external changes to the memory of arrays cannot be reflected in gradients.

Let's demonstrate this in an example:

.. code-block:: python

  def f(x):
      x_view = np.array(x, copy=False)
      x_view[:] *= x_view # modify memory without telling tk
      return x.sum()

  x = tk.array([3.0])
  y, df = tk.value_and_grad(f)(x)
  print("f(x) = x² =", y.item()) # 9.0
  print("f'(x) = 2x !=", df.item()) # 1.0


The function ``f`` indirectly modifies the array ``x`` through a memory view.
However, this modification is not reflected in the gradient, as seen in the
last line outputting ``1.0``, representing the gradient of the sum operation
alone.  The squaring of ``x`` occurs externally to Tiki, meaning that no
gradient is incorporated.  It's important to note that a similar issue arises
during array conversion and copying.  For instance, a function defined as
``tk.array(np.array(x)**2).sum()`` would also result in an incorrect gradient,
even though no in-place operations on Tiki memory are executed.

PyTorch
-------

PyTorch supports DLPack inputs and can import Tiki arrays directly.
Tiki can also import PyTorch tensors through DLPack with ``tk.asarray`` or
``tk.from_dlpack``. Use ``torch.as_tensor`` to import an Tiki array with
DLPack; ``torch.tensor`` copies the data instead. Similarly, ``tk.asarray``
can share DLPack inputs when possible, while ``tk.array`` copies:

.. code-block:: python

  import tiki as tk
  import torch

  a = tk.arange(3, dtype=tk.float32)
  tk.eval(a)

  shared = torch.as_tensor(a)
  copied = torch.tensor(a)

Creating an Tiki array from a CPU tensor copies the data into Tiki-owned storage.
The arrays do not share memory:

.. code-block:: python

  b = torch.arange(3)
  c = tk.array(b)

  b += 10
  print(c.tolist()) # [0, 1, 2]

Metal DLPack inputs are different. If a PyTorch MPS tensor is passed to
``tk.asarray`` or to ``tk.from_dlpack`` with ``copy=None``, Tiki imports it
without a copy when the underlying Metal buffer is not private. Private Metal
buffers are copied into Tiki-managed storage instead. Passing ``copy=False``
requires zero-copy import and raises an error if a copy would be needed.
Passing ``copy=True`` asks Tiki to create a new array instead of reusing the
Metal buffer. Zero-copy imports preserve the DLPack strides. ``tk.array`` also
creates a new array instead of reusing the Metal buffer. Tiki arrays exported to
PyTorch with DLPack are exported without a copy on Metal.

In particular, PyTorch 2.12 and later use shared storage for ordinary MPS
tensors on Apple silicon, while older PyTorch versions may use private storage
and require a copy on import. DLPack conversion does not synchronize pending
Metal work; synchronize or evaluate the producing framework before reading the
converted array.

.. code-block:: python

  b = torch.arange(3, device="mps", dtype=torch.float32)
  torch.mps.synchronize()
  c = tk.asarray(b) # zero-copy if the Metal buffer can be reused
  d = tk.from_dlpack(b, copy=True) # explicit copy

.. code-block:: python

  a = tk.arange(3, dtype=tk.float32)
  tk.eval(a)
  b = torch.as_tensor(a) # zero-copy DLPack import on Metal

JAX
---
JAX fully supports the buffer protocol.

.. code-block:: python

  import tiki as tk
  import jax.numpy as jnp

  a = tk.arange(3)
  b = jnp.array(a)
  c = tk.array(b)

TensorFlow
----------

TensorFlow supports the buffer protocol, but it requires an explicit
:obj:`memoryview`.

.. code-block:: python

  import tiki as tk
  import tensorflow as tf

  a = tk.arange(3)
  b = tf.constant(memoryview(a))
  c = tk.array(b)
