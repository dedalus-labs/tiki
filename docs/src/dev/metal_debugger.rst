Metal Debugger
==============

.. currentmodule:: tiki

Profiling is a key step for performance optimization. You can build Tiki with
the ``TIKI_METAL_DEBUG`` option to improve the Metal debugging and
optimization workflow. The ``TIKI_METAL_DEBUG`` debug option:

* Records source during Metal compilation, for later inspection while
  debugging.
* Labels Metal objects such as command queues, improving capture readability.

To build with debugging enabled in Python prepend
``CMAKE_ARGS="-DTIKI_METAL_DEBUG=ON"`` to the build call.

The :func:`metal.start_capture` function initiates a capture of all Tiki GPU
work.

.. note::

   To capture a GPU trace you must run the application with
   ``MTL_CAPTURE_ENABLED=1``.

.. code-block:: python

    import tiki as tk

    a = tk.random.uniform(shape=(512, 512))
    b = tk.random.uniform(shape=(512, 512))
    tk.eval(a, b)

    trace_file = "tiki_trace.gputrace"

    # Make sure to run with MTL_CAPTURE_ENABLED=1 and
    # that the path trace_file does not already exist.
    tk.metal.start_capture(trace_file)

    for _ in range(10):
      tk.eval(tk.add(a, b))

    tk.metal.stop_capture()

You can open and replay the GPU trace in Xcode. The ``Dependencies`` view
has a great overview of all operations. Checkout the `Metal debugger
documentation`_ for more information.

.. image:: ../_static/metal_debugger/capture.png
    :class: dark-light

Xcode Workflow
--------------

You can skip saving to a path by running within Xcode. First, generate an
Xcode project using CMake.

.. code-block::

    mkdir build && cd build
    cmake .. -DTIKI_METAL_DEBUG=ON -G Xcode
    open tiki.xcodeproj

Select the ``metal_capture`` example schema and run.

.. image:: ../_static/metal_debugger/schema.png
    :class: dark-light

.. _`Metal debugger documentation`: https://developer.apple.com/documentation/xcode/metal-debugger
