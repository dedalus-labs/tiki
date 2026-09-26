.. _export_usage:

Exporting Functions
===================

.. currentmodule:: tiki

Tiki has an API to export and import functions to and from a file. This lets you
run computations written in one Tiki front-end (e.g. Python) in another Tiki
front-end (e.g. C++).

This guide walks through the basics of the Tiki export API with some examples.
To see the full list of functions check-out the :ref:`API documentation
<export>`.

Basics of Exporting
-------------------

Let's start with a simple example:

.. code-block:: python

  def fun(x, y):
    return x + y

  x = tk.array(1.0)
  y = tk.array(1.0)
  tk.export_function("add.tkfn", fun, x, y)

To export a function, provide sample input arrays that the function
can be called with. The data doesn't matter, but the shapes and types of the
arrays do. In the above example we exported ``fun`` with two ``float32``
scalar arrays. We can then import the function and run it:

.. code-block:: python

  add_fun = tk.import_function("add.tkfn")

  out, = add_fun(tk.array(1.0), tk.array(2.0))
  # Prints: array(3, dtype=float32)
  print(out)

  out, = add_fun(tk.array(1.0), tk.array(3.0))
  # Prints: array(4, dtype=float32)
  print(out)

  # Raises an exception
  add_fun(tk.array(1), tk.array(3.0))

  # Raises an exception
  add_fun(tk.array([1.0, 2.0]), tk.array(3.0))

Notice the third and fourth calls to ``add_fun`` raise exceptions because the
shapes and types of the inputs are different than the shapes and types of the
example inputs we exported the function with.

Also notice that even though the original ``fun`` returns a single output
array, the imported function always returns a tuple of one or more arrays.

The inputs to :func:`export_function` and to an imported function can be
specified as variable positional arguments or as a tuple of arrays:

.. code-block:: python

  def fun(x, y):
    return x + y

  x = tk.array(1.0)
  y = tk.array(1.0)

  # Both arguments to fun are positional
  tk.export_function("add.tkfn", fun, x, y)

  # Same as above
  tk.export_function("add.tkfn", fun, (x, y))

  imported_fun = tk.import_function("add.tkfn")

  # Ok
  out, = imported_fun(x, y)

  # Also ok
  out, = imported_fun((x, y))

You can pass example inputs to functions as positional or keyword arguments. If
you use keyword arguments to export the function, then you have to use the same
keyword arguments when calling the imported function.

.. code-block:: python

  def fun(x, y):
    return x + y

  # One argument to fun is positional, the other is a kwarg
  tk.export_function("add.tkfn", fun, x, y=y)

  imported_fun = tk.import_function("add.tkfn")

  # Ok
  out, = imported_fun(x, y=y)

  # Also ok
  out, = imported_fun((x,), {"y": y})

  # Raises since the keyword argument is missing
  out, = imported_fun(x, y)

  # Raises since the keyword argument has the wrong key
  out, = imported_fun(x, z=y)


Saving Metadata
---------------

You can save metadata, such as a model configuration, alongside an exported
function. The metadata is a string, so structured data can be encoded with
JSON:

.. code-block:: python

  import json

  def fun(x, y):
    return x + y

  x = tk.array(1.0)
  y = tk.array(1.0)
  config = {"description": "adds two arrays", "version": 1}
  tk.export_function("add.tkfn", fun, x, y, metadata=json.dumps(config))

Pass ``return_metadata=True`` to read the metadata back when importing:

.. code-block:: python

  imported_fun, metadata = tk.import_function("add.tkfn", return_metadata=True)

  # Prints: adds two arrays
  print(json.loads(metadata)["description"])


Exporting Modules
-----------------

An :obj:`tiki.nn.Module` can be exported with or without the parameters included
in the exported function. Here's an example:

.. code-block:: python

   model = nn.Linear(4, 4)
   tk.eval(model.parameters())

   def call(x):
      return model(x)

   tk.export_function("model.tkfn", call, tk.zeros(4))

In the above example, the :obj:`tiki.nn.Linear` module is exported. Its
parameters are also saved to the ``model.tkfn`` file.

.. note::

   For enclosed arrays inside an exported function, be extra careful to ensure
   they are evaluated. The computation graph that gets exported will include
   the computation that produces enclosed inputs.

   If the above example was missing ``tk.eval(model.parameters()``, the
   exported function would include the random initialization of the
   :obj:`tiki.nn.Module` parameters.

If you only want to export the ``Module.__call__`` function without the
parameters, pass them as inputs to the ``call`` wrapper:

.. code-block:: python

   model = nn.Linear(4, 4)
   tk.eval(model.parameters())

   def call(x, **params):
     # Set the model's parameters to the input parameters
     model.update(tree_unflatten(list(params.items())))
     return model(x)

   params = tree_flatten(model.parameters(), destination={})
   tk.export_function("model.tkfn", call, (tk.zeros(4),), params)


Exporting with a Callback
-------------------------

To inspect the exported graph, you can pass a callback instead of a file path
to :func:`export_function`.

.. code-block:: python

  def fun(x):
    return x.astype(tk.int32)

  def callback(args):
    print(args)

  tk.export_function(callback, fun, tk.array([1.0, 2.0]))

The argument to the callback (``args``) is a dictionary which includes a
``type`` field. The possible types are:

* ``"inputs"``: The ordered positional inputs to the exported function
* ``"keyword_inputs"``: The keyword specified inputs to the exported function
* ``"outputs"``: The ordered outputs of the exported function
* ``"constants"``: Any graph constants
* ``"primitives"``: Inner graph nodes representating the operations

Each type has additional fields in the ``args`` dictionary.


Shapeless Exports
-----------------

Just like :func:`compile`, functions can also be exported for dynamically shaped
inputs. Pass ``shapeless=True`` to :func:`export_function` or :func:`exporter`
to export a function which can be used for inputs with variable shapes:

.. code-block:: python

  tk.export_function("fun.tkfn", tk.abs, tk.array([0.0]), shapeless=True)
  imported_abs = tk.import_function("fun.tkfn")

  # Ok
  out, = imported_abs(tk.array([-1.0]))

  # Also ok
  out, = imported_abs(tk.array([-1.0, -2.0]))

With ``shapeless=False`` (which is the default), the second call to
``imported_abs`` would raise an exception with a shape mismatch.

Shapeless exporting works the same as shapeless compilation and should be
used carefully. See the :ref:`documentation on shapeless compilation
<shapeless_compile>` for more information.

Exporting Multiple Traces
-------------------------

In some cases, functions build different computation graphs for different
input arguments. A simple way to manage this is to export to a new file with
each set of inputs. This is a fine option in many cases. But it can be
suboptimal if the exported functions have a large amount of duplicate constant
data (for example the parameters of a :obj:`tiki.nn.Module`).

The export API in Tiki lets you export multiple traces of the same function to
a single file by creating an exporting context manager with :func:`exporter`:

.. code-block:: python

  def fun(x, y=None):
      constant = tk.array(3.0)
      if y is not None:
        x += y
      return x + constant

  with tk.exporter("fun.tkfn", fun) as exporter:
      exporter(tk.array(1.0))
      exporter(tk.array(1.0), y=tk.array(0.0))

  imported_function = tk.import_function("fun.tkfn")

  # Call the function with y=None
  out, = imported_function(tk.array(1.0))
  print(out)

  # Call the function with y specified
  out, = imported_function(tk.array(1.0), y=tk.array(1.0))
  print(out)

In the above example the function constant data, (i.e. ``constant``), is only
saved once.

Transformations with Imported Functions
---------------------------------------

Function transformations like :func:`grad`, :func:`vmap`, and :func:`compile` work
on imported functions just like regular Python functions:

.. code-block:: python

  def fun(x):
      return tk.sin(x)

  x = tk.array(0.0)
  tk.export_function("sine.tkfn", fun, x)

  imported_fun = tk.import_function("sine.tkfn")

  # Take the derivative of the imported function
  dfdx = tk.grad(lambda x: imported_fun(x)[0])
  # Prints: array(1, dtype=float32)
  print(dfdx(x))

  # Compile the imported function
  tk.compile(imported_fun)
  # Prints: array(0, dtype=float32)
  print(compiled_fun(x)[0])


Importing Functions in C++
--------------------------

Importing and running functions in C++ is basically the same as importing and
running them in Python. First, follow the :ref:`instructions <tiki_in_cpp>` to
setup a simple C++ project that uses Tiki as a library.

Next, export a simple function from Python:

.. code-block:: python

  def fun(x, y):
      return tk.exp(x + y)

  x = tk.array(1.0)
  y = tk.array(1.0)
  tk.export_function("fun.tkfn", fun, x, y)


Import and run the function in C++ with only a few lines of code:

.. code-block:: c++

  auto fun = tk::import_function("fun.tkfn");

  auto inputs = {tk::array(1.0), tk::array(1.0)};
  auto outputs = fun(inputs);

  // Prints: array(2, dtype=float32)
  std::cout << outputs[0] << std::endl;

Imported functions can be transformed in C++ just like in Python. Use
``std::vector<tk::array>`` for positional arguments and ``std::map<std::string,
tk::array>`` for keyword arguments when calling imported functions in C++.

More Examples
-------------

Here are a few more complete examples exporting more complex functions from
Python and importing and running them in C++:

* `Inference and training a multi-layer perceptron <https://github.com/ml-explore/mlx/tree/main/examples/export>`_
