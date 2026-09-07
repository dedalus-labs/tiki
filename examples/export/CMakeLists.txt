cmake_minimum_required(VERSION 3.27)

project(import_tiki LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tiki --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TIKI_ROOT)
find_package(Tiki CONFIG REQUIRED)

add_executable(eval_mlp eval_mlp.cpp)
target_link_libraries(eval_mlp PRIVATE tiki)

add_executable(train_mlp train_mlp.cpp)
target_link_libraries(train_mlp PRIVATE tiki)
