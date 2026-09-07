cmake_minimum_required(VERSION 3.27)

project(example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Comment the following two commands only the Tiki C++ library is installed and
# set(TIKI_ROOT "/path/to/tiki") directly if needed.
find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tiki --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TIKI_ROOT)

find_package(Tiki CONFIG REQUIRED)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE tiki)
