cmake_minimum_required(VERSION 3.25)

if(NOT TIKI_VERSION)
  file(STRINGS "tiki/version.h" _tiki_h_version
       REGEX "^#define TIKI_VERSION_.*$")
  string(REGEX MATCH "#define TIKI_VERSION_MAJOR ([0-9]+)" _
               "${_tiki_h_version}")
  set(_major ${CMAKE_MATCH_1})
  string(REGEX MATCH "#define TIKI_VERSION_MINOR ([0-9]+)" _
               "${_tiki_h_version}")
  set(_minor ${CMAKE_MATCH_1})
  string(REGEX MATCH "#define TIKI_VERSION_PATCH ([0-9]+)" _
               "${_tiki_h_version}")
  set(_patch ${CMAKE_MATCH_1})
  set(TIKI_PROJECT_VERSION "${_major}.${_minor}.${_patch}")
  set(TIKI_VERSION ${TIKI_PROJECT_VERSION})
else()
  string(REGEX REPLACE "^([0-9]+\.[0-9]+\.[0-9]+).*" "\\1" TIKI_PROJECT_VERSION
                       ${TIKI_VERSION})
endif()

project(
  tiki
  LANGUAGES C CXX
  VERSION ${TIKI_PROJECT_VERSION})

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------- Configuration -----------------------------
option(TIKI_BUILD_TESTS "Build tests for tiki" ON)
option(TIKI_BUILD_EXAMPLES "Build examples for tiki" ON)
option(TIKI_BUILD_BENCHMARKS "Build benchmarks for tiki" OFF)
option(TIKI_BUILD_PYTHON_BINDINGS "Build python bindings for tiki" OFF)
option(TIKI_BUILD_METAL "Build metal backend" ON)
option(TIKI_BUILD_CPU "Build cpu backend" ON)
option(TIKI_BUILD_CUDA "Build cuda backend" OFF)
option(TIKI_METAL_DEBUG "Enhance metal debug workflow" OFF)
option(TIKI_ENABLE_X64_MAC "Enable building for x64 macOS" OFF)
option(TIKI_BUILD_GGUF "Include support for GGUF format" ON)
option(TIKI_BUILD_SAFETENSORS "Include support for safetensors format" ON)
option(TIKI_BUILD_PYTHON_STUBS "Build stub files for python bindings" ON)
option(TIKI_METAL_JIT "Use JIT compilation for Metal kernels" OFF)
option(TIKI_USE_CCACHE "Use CCache for compilation cache when available" ON)
option(TIKI_EXPORT_ALL_SYMBOLS "Export all symbols in the library" OFF)
option(BUILD_SHARED_LIBS "Build tiki as a shared library" OFF)
option(USE_SYSTEM_FMT "Use system's provided fmt library" OFF)
option(USE_ASAN "Enable AddressSanitizer (ASan)" OFF)
option(USE_UBSAN "Enable UndefinedBehaviorSanitizer (UBSan)" OFF)
option(USE_TSAN "Enable ThreadSanitizer (TSan)" OFF)

# --------------------- Processor tests -------------------------
message(
  STATUS
    "Building Tiki for ${CMAKE_SYSTEM_PROCESSOR} processor on ${CMAKE_SYSTEM_NAME}"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    if(NOT TIKI_ENABLE_X64_MAC)
      message(
        FATAL_ERROR
          "Building for x86_64 on macOS is not supported."
          " If you are on an Apple silicon system, check the build"
          " documentation for possible fixes: "
          "https://ml-explore.github.io/tiki/build/html/install.html#build-from-source"
      )
    else()
      set(TIKI_BUILD_METAL OFF)
      message(WARNING "Building for x86_64 arch is not officially supported.")
    endif()
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  # Enable the Metal backend by default on iOS; the caller selects the SDK.
else()
  set(TIKI_BUILD_METAL OFF)
endif()

if(TIKI_USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()

if(USE_ASAN AND USE_TSAN)
  message(
    FATAL_ERROR
      "AddressSanitizer (ASan) and ThreadSanitizer (TSan) are mutually exclusive and cannot be enabled at the same time."
  )
endif()

set(SANITIZER_COMPILE_FLAGS "")
set(SANITIZER_LINK_FLAGS "")

if(USE_ASAN)
  if(WIN32 AND MSVC)
    list(APPEND SANITIZER_COMPILE_FLAGS /fsanitize=address)
    list(APPEND SANITIZER_LINK_FLAGS /fsanitize=address)
  else()
    list(APPEND SANITIZER_COMPILE_FLAGS -fsanitize=address)
    list(APPEND SANITIZER_LINK_FLAGS -fsanitize=address)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND SANITIZER_LINK_FLAGS -lpthread)
    endif()
  endif()
endif()

if(USE_UBSAN)
  if(WIN32 AND MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      list(APPEND SANITIZER_COMPILE_FLAGS -fsanitize=undefined)
      list(APPEND SANITIZER_LINK_FLAGS -fsanitize=undefined)
    else()
      message(
        WARNING
          "UndefinedBehaviorSanitizer (UBSan) is not directly supported via a simple flag in MSVC."
      )
    endif()
  else()
    list(APPEND SANITIZER_COMPILE_FLAGS -fsanitize=undefined)
    list(APPEND SANITIZER_LINK_FLAGS -fsanitize=undefined)
  endif()
endif()

if(USE_TSAN)
  if(WIN32 AND MSVC)
    message(
      FATAL_ERROR
        "ThreadSanitizer (TSan) is not supported by the MSVC compiler. Please use Clang or GCC."
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(FATAL_ERROR "ThreadSanitizer (TSan) is not supported on macOS.")
  else()
    list(APPEND SANITIZER_COMPILE_FLAGS -fsanitize=thread)
    list(APPEND SANITIZER_LINK_FLAGS -fsanitize=thread)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND SANITIZER_LINK_FLAGS -lpthread)
    endif()
  endif()
endif()

# ----------------------------- Lib -----------------------------

include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
cmake_policy(SET CMP0135 NEW)

add_library(tiki)

target_compile_features(tiki INTERFACE cxx_std_20)
target_compile_options(tiki PUBLIC ${SANITIZER_COMPILE_FLAGS})
target_link_options(tiki PUBLIC ${SANITIZER_LINK_FLAGS})

if(TIKI_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(CUDNN REQUIRED)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.1" AND CUDAToolkit_VERSION
                                                          VERSION_LESS "13.2")
    message(FATAL_ERROR "CUDA Toolkit 13.1 is not supported.")
  endif()
endif()

if(TIKI_BUILD_METAL)
  find_library(METAL_LIB Metal)
  find_library(FOUNDATION_LIB Foundation)
  find_library(QUARTZ_LIB QuartzCore)
  if(METAL_LIB)
    message(STATUS "Metal found ${METAL_LIB}")
  else()
    message(
      FATAL_ERROR
        "Metal not found. Set TIKI_BUILD_METAL=OFF to build without GPU")
  endif()

  if(TIKI_METAL_DEBUG)
    add_compile_definitions(TIKI_METAL_DEBUG)
  endif()

  # Probe the macOS SDK only on Darwin. On iOS the caller selects the SDK, so
  # version detection is deferred and the baseline Metal feature set is used.
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # Get the host macOS version.
    execute_process(
      COMMAND sw_vers -productVersion
      OUTPUT_VARIABLE MACOS_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    message(STATUS "Building on macOS ${MACOS_VERSION}")

    # Get macOS SDK version.
    execute_process(
      COMMAND xcrun -sdk macosx --show-sdk-version
      OUTPUT_VARIABLE MACOS_SDK_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    if(MACOS_SDK_VERSION VERSION_LESS 14.0)
      message(
        FATAL_ERROR
          "Tiki requires macOS SDK >= 14.0 to be built with TIKI_BUILD_METAL=ON"
      )
    endif()
    message(STATUS "Building with macOS ${MACOS_SDK_VERSION} SDK")

    # Build for the host system if CMAKE_OSX_DEPLOYMENT_TARGET is not set.
    if(CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
      set(CMAKE_OSX_DEPLOYMENT_TARGET "${MACOS_VERSION}")
    endif()
    if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 14.0)
      message(FATAL_ERROR "Tiki requires macOS >= 14.0")
    endif()
    message(STATUS "Building for macOS ${CMAKE_OSX_DEPLOYMENT_TARGET}")

    # Get Metal version.
    set(XCRUN_FLAGS "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    execute_process(
      COMMAND
        zsh "-c"
        "echo \"__METAL_VERSION__\" | xcrun -sdk macosx metal ${XCRUN_FLAGS} -E -x metal -P - | tail -1"
      OUTPUT_VARIABLE TIKI_METAL_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    message(STATUS "Metal version: ${TIKI_METAL_VERSION}")
  else()
    set(TIKI_METAL_VERSION 0)
  endif()

  set(METAL_CPP_URL
      https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip)
  FetchContent_Declare(metal_cpp URL ${METAL_CPP_URL})
  FetchContent_MakeAvailable(metal_cpp)
  target_include_directories(
    tiki PUBLIC $<BUILD_INTERFACE:${metal_cpp_SOURCE_DIR}>
                $<INSTALL_INTERFACE:include/metal_cpp>)
  target_link_libraries(tiki PUBLIC ${METAL_LIB} ${FOUNDATION_LIB}
                                    ${QUARTZ_LIB})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # With newer clang/gcc versions following libs are implicitly linked, but when
  # building on old distributions they need to be explicitly listed.
  target_link_libraries(tiki PRIVATE dl pthread)
endif()

if(WIN32)
  if(MSVC)
    # GGUF does not build with MSVC.
    set(TIKI_BUILD_GGUF OFF)
  endif()
  # Generate DLL and EXE in the same dir, otherwise EXE will not be able to run.
  # This is only done when Tiki is built as the top project.
  if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()
  # Windows implementation of dlfcn.h APIs.
  FetchContent_Declare(
    dlfcn-win32
    GIT_REPOSITORY https://github.com/dlfcn-win32/dlfcn-win32.git
    GIT_TAG v1.4.2
    EXCLUDE_FROM_ALL)
  block()
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_MakeAvailable(dlfcn-win32)
  endblock()
  target_include_directories(tiki PRIVATE "${dlfcn-win32_SOURCE_DIR}/src")
  target_sources(tiki PRIVATE "${dlfcn-win32_SOURCE_DIR}/src/dlfcn.c")
  target_link_libraries(tiki PRIVATE Psapi.lib)
endif()

if(TIKI_BUILD_CPU)
  find_library(ACCELERATE_LIBRARY Accelerate)
  if(ACCELERATE_LIBRARY)
    message(STATUS "Accelerate found ${ACCELERATE_LIBRARY}")
    set(TIKI_BUILD_ACCELERATE ON)
  else()
    message(STATUS "Accelerate not found, using default backend.")
    set(TIKI_BUILD_ACCELERATE OFF)
  endif()

  if(TIKI_BUILD_ACCELERATE)
    target_link_libraries(tiki PUBLIC ${ACCELERATE_LIBRARY})
    add_compile_definitions(TIKI_USE_ACCELERATE)
    add_compile_definitions(ACCELERATE_NEW_LAPACK)
  elseif(WIN32)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
      set(OPENBLAS_ZIP "OpenBLAS-0.3.33-woa64-dll.zip")
      set(OPENBLAS_LIB "openblas")
      set(OPENBLAS_INC "include/openblas")
    else()
      set(OPENBLAS_ZIP "OpenBLAS-0.3.33-x64.zip")
      set(OPENBLAS_LIB "libopenblas")
      set(OPENBLAS_INC "include")
    endif()
    # Download and link prebuilt binaries of OpenBLAS. Note that we can only
    # link with the dynamic library, the prebuilt binaries were built with MinGW
    # so static-linking would require linking with MinGW's runtime.
    FetchContent_Declare(
      openblas
      URL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.33/${OPENBLAS_ZIP}"
    )
    FetchContent_MakeAvailable(openblas)
    target_include_directories(tiki
                               PRIVATE "${openblas_SOURCE_DIR}/${OPENBLAS_INC}")
    # Make openblas importable by dependencies when built as static library.
    set(OPENBLAS_IMPORT_LIBRARY
        "${openblas_SOURCE_DIR}/lib/${OPENBLAS_LIB}.lib")
    add_library(TIKI::OpenBLAS UNKNOWN IMPORTED GLOBAL)
    set_target_properties(
      TIKI::OpenBLAS PROPERTIES IMPORTED_LOCATION "${OPENBLAS_IMPORT_LIBRARY}")
    target_link_libraries(tiki PRIVATE TIKI::OpenBLAS)
    # Make sure the DLL file is placed in the same dir with executables.
    set(OPENBLAS_DLL_FILE "${openblas_SOURCE_DIR}/bin/${OPENBLAS_LIB}.dll")
    add_custom_command(
      TARGET tiki
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENBLAS_DLL_FILE}
              ${CMAKE_BINARY_DIR})
  else()
    if(${CMAKE_HOST_APPLE})
      # The blas shipped in macOS SDK is not supported, search homebrew for
      # openblas instead.
      set(BLA_VENDOR OpenBLAS)
      set(LAPACK_ROOT
          "${LAPACK_ROOT};$ENV{LAPACK_ROOT};/usr/local/opt/openblas")
    endif()
    # Search and link with lapack.
    find_package(LAPACK REQUIRED)
    if(NOT LAPACK_FOUND)
      message(FATAL_ERROR "Must have LAPACK installed")
    endif()
    find_path(LAPACK_INCLUDE_DIRS lapacke.h /usr/include /usr/local/include
              /usr/local/opt/openblas/include)
    message(STATUS "Lapack lib " ${LAPACK_LIBRARIES})
    message(STATUS "Lapack include " ${LAPACK_INCLUDE_DIRS})
    target_include_directories(tiki PRIVATE ${LAPACK_INCLUDE_DIRS})
    target_link_libraries(tiki PRIVATE ${LAPACK_LIBRARIES})
    # List blas after lapack otherwise we may accidentally incldue an old
    # version of lapack.h from the include dirs of blas.
    find_package(BLAS REQUIRED)
    if(NOT BLAS_FOUND)
      message(FATAL_ERROR "Must have BLAS installed")
    endif()
    # TODO find a cleaner way to do this
    find_path(BLAS_INCLUDE_DIRS cblas.h /usr/include /usr/local/include
              $ENV{BLAS_HOME}/include)
    message(STATUS "Blas lib " ${BLAS_LIBRARIES})
    message(STATUS "Blas include " ${BLAS_INCLUDE_DIRS})
    target_include_directories(tiki PRIVATE ${BLAS_INCLUDE_DIRS})
    target_link_libraries(tiki PRIVATE ${BLAS_LIBRARIES})
  endif()
else()
  set(TIKI_BUILD_ACCELERATE OFF)
endif()

message(STATUS "Downloading json")
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_include_directories(
  tiki PRIVATE $<BUILD_INTERFACE:${json_SOURCE_DIR}/single_include/nlohmann>)

# Add standalone JACCL library (RDMA over Thunderbolt distributed backend)
if(TIKI_BUILD_CPU
   AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin"
   AND DEFINED MACOS_SDK_VERSION
   AND MACOS_SDK_VERSION VERSION_GREATER_EQUAL 26.2)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tiki/distributed/jaccl/lib
                   ${CMAKE_BINARY_DIR}/jaccl)
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tiki)

target_include_directories(
  tiki PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
              $<INSTALL_INTERFACE:include>)

if(USE_SYSTEM_FMT)
  find_package(fmt REQUIRED)
else()
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.1.0
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(fmt)
endif()
target_link_libraries(tiki PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)

if(TIKI_BUILD_PYTHON_BINDINGS)
  message(STATUS "Building Python bindings.")
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
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/python/src)
endif()

if(TIKI_BUILD_TESTS)
  include(CTest)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
endif()

if(TIKI_BUILD_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/examples/cpp)
endif()

if(TIKI_BUILD_BENCHMARKS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/benchmarks/cpp)
endif()

# ----------------------------- Installation -----------------------------
include(GNUInstallDirs)

if(WIN32)
  # Install DLLs to the same dir with extension file (core.pyd) on Windows.
  set(CMAKE_INSTALL_BINDIR ".")
  if(TIKI_BUILD_CPU)
    # Install OpenBLAS.
    install(FILES ${OPENBLAS_DLL_FILE} TYPE BIN)
    install(FILES ${OPENBLAS_IMPORT_LIBRARY}
            DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
endif()

# Install library
install(
  TARGETS tiki
  EXPORT TIKITargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install headers
install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tiki
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT headers
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "backend/metal/kernels.h" EXCLUDE)

# Install metal dependencies
if(TIKI_BUILD_METAL)

  # Install metal cpp
  install(
    DIRECTORY ${metal_cpp_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metal_cpp
    COMPONENT metal_cpp_source)

endif()

# Install cmake config
set(TIKI_CMAKE_BUILD_CONFIG ${CMAKE_BINARY_DIR}/TIKIConfig.cmake)
set(TIKI_CMAKE_BUILD_VERSION_CONFIG ${CMAKE_BINARY_DIR}/TIKIConfigVersion.cmake)
set(TIKI_CMAKE_INSTALL_MODULE_DIR share/cmake/Tiki)

install(
  EXPORT TIKITargets
  FILE TIKITargets.cmake
  DESTINATION ${TIKI_CMAKE_INSTALL_MODULE_DIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${TIKI_CMAKE_BUILD_VERSION_CONFIG}
  COMPATIBILITY SameMajorVersion
  VERSION ${TIKI_VERSION})

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/tiki.pc.in ${TIKI_CMAKE_BUILD_CONFIG}
  INSTALL_DESTINATION ${TIKI_CMAKE_INSTALL_MODULE_DIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
            TIKI_CMAKE_INSTALL_MODULE_DIR)

install(FILES ${TIKI_CMAKE_BUILD_CONFIG} ${TIKI_CMAKE_BUILD_VERSION_CONFIG}
        DESTINATION ${TIKI_CMAKE_INSTALL_MODULE_DIR})

install(DIRECTORY ${CMAKE_MODULE_PATH}/
        DESTINATION ${TIKI_CMAKE_INSTALL_MODULE_DIR})
