.. _tiki-layout-build:

Building and testing Tiki layouts
=================================

Build the framework
-------------------

Install the usual Tiki prerequisites and Rust 1.92 or newer. A normal
``python -m pip install .`` builds the Rust indexing extension automatically.
The extension uses CXX and owns no CUDA resources. Affine reference algebra
remains in PyCuTe. Rust's ``lib.rs`` is an index, not an implementation module.

With Tiki already installed, build only the indexing extension for development:

.. code-block:: sh

   cmake -S . -B build/indexing \
     -DTIKI_BUILD_CPU=OFF -DTIKI_BUILD_METAL=OFF -DTIKI_BUILD_CUDA=OFF \
     -DTIKI_BUILD_TESTS=OFF -DTIKI_BUILD_EXAMPLES=OFF \
     -DTIKI_BUILD_PYTHON_BINDINGS=ON -DTIKI_BUILD_PYTHON_STUBS=OFF \
     -DPython_EXECUTABLE="$(command -v python)" \
     -DTIKI_PYTHON_BINDINGS_OUTPUT_DIRECTORY="$PWD/python/tiki"
   cmake --build build/indexing --target tiki_layout_python

This target does not rebuild ``tiki``. Add ``python/`` to ``PYTHONPATH``
to use the checkout alongside that core. On macOS, set
``CMAKE_OSX_DEPLOYMENT_TARGET`` to the core's minimum OS. CMake forwards it to Cargo.

Verify code and examples
------------------------

.. code-block:: sh

   cargo test --manifest-path tiki/layout/Cargo.toml --all-features
   cargo fmt --manifest-path tiki/layout/Cargo.toml --check
   cargo clippy --manifest-path tiki/layout/Cargo.toml --all-targets --all-features -- -D warnings
   PYTHONPATH=python:python/tests python -m unittest discover -s python/tests -p 'test_tiki_*.py'

Install Doxygen and ``docs/requirements.txt``. With a Tiki core on the Python
path, build the site and test its explicit doctest directives:

.. code-block:: sh

   cd docs
   doxygen
   PYTHONPATH=../python make html O=-W
   PYTHONPATH=../python sphinx-build -b doctest -W \
     -D doctest_test_doctest_blocks= src build/doctest

The site is in ``docs/build/html``. ``test_tiki_docs`` also reads the guide and
recipes directly, so the examples are tested without maintaining separate copies.
These checks do not establish CUDA memory safety or general nonlinear kernel
lowering. See :ref:`tiki-layouts` for the compiler integration boundary.
