.. _precision:

Numerical Precision
===================

``float32`` matrix-multiplication family operations (matmul, quantized
matmul, grouped matmul, convolution and attention) run in full ``float32``
precision. Hardware with dedicated matrix-multiplication units can run them
at reduced precision instead: inputs and outputs stay ``float32``, but
results can differ from a full-precision reference by several orders of
magnitude more than ``float32`` rounding alone would explain.

To opt in to the reduced-precision path, set :envvar:`MLX_ENABLE_TF32` to
``1`` when launching the process:

.. code-block:: shell

  MLX_ENABLE_TF32=1 python my_script.py

Which operations take the reduced-precision path, and how large the
difference is, depends on the backend and the hardware.
