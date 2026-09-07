.. _tiki_in_cpp:

Using Tiki in C++
=================

You can use Tiki in a C++ project with CMake.

.. note::

  This guide is based one the following `example using Tiki in C++ 
  <https://github.com/ml-explore/mlx/tree/main/examples/cmake_project>`_

First install Tiki:

.. code-block:: bash

  pip install -U tiki

You can also install the Tiki Python package from source or just the C++
library. For more information see the :ref:`documentation on installing Tiki
<build_and_install>`.

Next make an example program in ``example.cpp``: 

.. code-block:: C++

  #include <iostream>

  #include "tiki/tiki.h"

  namespace tk = tiki::core;

  int main() {
    auto x = tk::array({1, 2, 3});
    auto y = tk::array({1, 2, 3});
    std::cout << x + y << std::endl;
    return 0;
  }

The next step is to setup a CMake file in ``CMakeLists.txt``:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.27)

  project(example LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)


Depending on how you installed Tiki, you may need to tell CMake where to
find it. 

If you installed Tiki with Python, then add the following to the CMake file:

.. code-block:: cmake

  find_package(
    Python 3.9
    COMPONENTS Interpreter Development.Module
    REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tiki --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TIKI_ROOT)

If you installed the Tiki C++ package to a system path, then CMake should be
able to find it. If you installed it to a non-standard location or CMake can't
find Tiki then set ``TIKI_ROOT`` to the location where Tiki is installed:

.. code-block:: cmake

  set(TIKI_ROOT "/path/to/tiki/")

Next, instruct CMake to find Tiki:

.. code-block:: cmake

  find_package(Tiki CONFIG REQUIRED)

Finally, add the ``example.cpp`` program as an executable and link Tiki.

.. code-block:: cmake

  add_executable(example example.cpp)
  target_link_libraries(example PRIVATE tiki)

You can build the example with:

.. code-block:: bash

  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build

And run it with:

.. code-block:: bash

  ./build/example

Note ``find_package(Tiki CONFIG REQUIRED)`` sets the following variables:

.. list-table:: Package Variables
   :widths: 20 20 
   :header-rows: 1

   * - Variable 
     - Description 
   * - TIKI_FOUND
     - ``True`` if Tiki is found
   * - TIKI_INCLUDE_DIRS
     - Include directory
   * - TIKI_LIBRARIES
     - Libraries to link against
   * - TIKI_CXX_FLAGS
     - Additional compiler flags
   * - TIKI_BUILD_ACCELERATE
     - ``True`` if Tiki was built with Accelerate 
   * - TIKI_BUILD_METAL
     - ``True`` if Tiki was built with Metal
