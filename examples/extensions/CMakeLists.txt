cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------

include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
cmake_policy(SET CMP0135 NEW)

find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v3.0.1
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(nanobind)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tiki --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TIKI_ROOT)
find_package(Tiki CONFIG REQUIRED)

# ----------------------------- Extensions -----------------------------

# Add library
add_library(tiki_ext)

# Add sources
target_sources(tiki_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.cpp)

# Add include headers
target_include_directories(tiki_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Link to tiki
target_link_libraries(tiki_ext PUBLIC tiki)

# ----------------------------- Metal -----------------------------

# Build metallib
if(TIKI_BUILD_METAL)
  tiki_build_metallib(
    TARGET
    tiki_ext_metallib
    TITLE
    tiki_ext
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.metal
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${TIKI_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(tiki_ext tiki_ext_metallib)

endif()

# ----------------------------- Python Bindings -----------------------------
nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  tiki
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE tiki_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
