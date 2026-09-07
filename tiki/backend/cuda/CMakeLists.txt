# Filename rules in cuda backend:
#
# * Use .cu/.cuh if code contains device code, and .cpp/.h if not.
# * Device-only code should be put in device/ subdir.
# * Files in device/ subdir should not include files outside.
target_sources(
  tiki
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arange.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/binary_two.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/cholesky.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_contiguous.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_dynamic.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_input.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/conv/gemm_conv.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/conv/gemm_grouped_conv.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/cublas_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cross_entropy.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/cudnn_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cusolver_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/fft.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/block_mask.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/gemv.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_gemm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/gather_gemm.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/grouped_gemm_unaligned.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/hadamard.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/jit_module.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel_utils.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/random.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/all_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/col_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/init_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/row_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/rope.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/scaled_dot_product_attention.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scaled_dot_product_attention.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/scan.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/ternary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qmm/qmm.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qmm/qmm_naive.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qmm/qmm_sm80.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qmm/qmm_sm90.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qmm/qmv.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qmm/fp_qmv.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/affine_quantize.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/fp_quantize.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/quantized.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qqmm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qqmm_utils.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/convert_fp8.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp)

# Storage ownership lives in the Rust crate under runtime/. Cargo builds it and
# generates the CXX header that allocator.cpp includes.
find_program(CARGO cargo REQUIRED)
set(TIKI_CUDA_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
set(TIKI_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo)
set(TIKI_CUDA_RUNTIME_LIB
    ${TIKI_CARGO_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}tiki_cuda_runtime${CMAKE_STATIC_LIBRARY_SUFFIX}
)
file(GLOB_RECURSE TIKI_CUDA_RUNTIME_SOURCES ${TIKI_CUDA_RUNTIME_DIR}/src/*.rs)
add_custom_command(
  OUTPUT ${TIKI_CUDA_RUNTIME_LIB}
  COMMAND
    ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${TIKI_CARGO_TARGET_DIR} ${CARGO}
    build --locked --release --manifest-path ${TIKI_CUDA_RUNTIME_DIR}/Cargo.toml
  DEPENDS ${TIKI_CUDA_RUNTIME_SOURCES} ${TIKI_CUDA_RUNTIME_DIR}/Cargo.toml
          ${TIKI_CUDA_RUNTIME_DIR}/Cargo.lock ${TIKI_CUDA_RUNTIME_DIR}/build.rs
  COMMENT "Building tiki-cuda-runtime"
  VERBATIM)
add_custom_target(tiki_cuda_runtime DEPENDS ${TIKI_CUDA_RUNTIME_LIB})
add_dependencies(tiki tiki_cuda_runtime)
target_include_directories(tiki PRIVATE ${TIKI_CARGO_TARGET_DIR}/cxxbridge)
target_link_libraries(tiki PRIVATE ${TIKI_CUDA_RUNTIME_LIB} ${CMAKE_DL_LIBS})

# Put dynamic defines in the dirs.cpp file.
add_library(tiki_dirs OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/dirs.cpp)
target_include_directories(tiki_dirs PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(tiki PRIVATE $<BUILD_INTERFACE:tiki_dirs>)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/binary)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unary)

# fp4 is not available on < 12.8
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8.0)
  target_include_directories(tiki
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quantized/)
  target_sources(tiki
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quantized/no_qqmm_impl.cpp)
else()
  target_sources(
    tiki PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qqmm_impl.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/quantized/cublas_qqmm.cpp)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9.0)
  target_sources(
    tiki PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_gemm_batched_12_9.cu)
else()
  target_sources(
    tiki PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_gemm_batched_12_0.cpp)
endif()

# Embed kernel sources in binary for JIT compilation.
file(
  GLOB TIKI_JIT_SOURCES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.cuh")
string(JOIN ":" TIKI_JIT_SOURCES_ARG ${TIKI_JIT_SOURCES})
add_custom_command(
  OUTPUT gen/cuda_jit_sources.h
  COMMAND
    ${CMAKE_COMMAND} -DTIKI_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
    -DTIKI_JIT_SOURCES=${TIKI_JIT_SOURCES_ARG} -P
    "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
  DEPENDS bin2h.cmake ${TIKI_JIT_SOURCES})
add_custom_target(cuda_jit_sources DEPENDS gen/cuda_jit_sources.h)
add_dependencies(tiki cuda_jit_sources)
target_include_directories(tiki PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")

# ------------------------ Compilation configs ------------------------

target_compile_definitions(tiki PRIVATE TIKI_USE_CUDA)

# Enable defining device lambda functions.
target_compile_options(tiki
                       PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")

# Enable calling host constexpr functions from device. This is needed because
# the constexpr version of isnan is host only.
target_compile_options(
  tiki PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# Ignore warnings from CUTLASS.
target_compile_options(
  tiki
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=186,2908,2361">)

if(NOT MSVC)
  # Required for generating optimized CUTLASS code.
  target_compile_options(
    tiki PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-strict-aliasing>")
endif()

# Suppress nvcc warnings on C++ headers.
target_compile_options(
  tiki
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=27,997,1394,20011,20208">
)

# Ignore some valid nvcc warnings, we might want to fix them in future.
target_compile_options(
  tiki PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177,550">)

# Use stronger binaries compression. This feature was introduced in CUDA 12.8
# and requires drivers released after CUDA 12.4.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0)
  target_compile_options(
    tiki PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>")
endif()

# Use native CUDA arch by default.
if(NOT DEFINED TIKI_CUDA_ARCHITECTURES)
  execute_process(
    COMMAND __nvcc_device_query
    OUTPUT_VARIABLE TIKI_CUDA_ARCHITECTURES
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(TIKI_CUDA_ARCHITECTURES STREQUAL "")
    message(
      FATAL_ERROR
        "Can not get native CUDA arch, must set TIKI_CUDA_ARCHITECTURES")
  elseif(TIKI_CUDA_ARCHITECTURES GREATER_EQUAL 90)
    # Use arch-specific compute capability whenever possible.
    set(TIKI_CUDA_ARCHITECTURES "${TIKI_CUDA_ARCHITECTURES}a")
  endif()
endif()
message(STATUS "CUDA architectures: ${TIKI_CUDA_ARCHITECTURES}")
set_target_properties(tiki PROPERTIES CUDA_ARCHITECTURES
                                      "${TIKI_CUDA_ARCHITECTURES}")

# Configure Windows CUDA DLL loading.
if(WIN32)
  set(TIKI_CUDA_BIN_DIR
      ""
      CACHE STRING "Directory containing CUDA DLLs for Windows delay-loading")
  set(TIKI_CUDNN_BIN_DIR
      ""
      CACHE STRING "Directory containing cuDNN DLLs for Windows delay-loading")

  # With TIKI_LOAD_CUDA_LIBS_FROM_PYTHON, unset dirs use the wheel layout.
  # Relative dirs are resolved from the Tiki binary.
  if(NOT TIKI_LOAD_CUDA_LIBS_FROM_PYTHON)
    if("${TIKI_CUDA_BIN_DIR}" STREQUAL "")
      set(TIKI_CUDA_BIN_DIR "${CUDAToolkit_BIN_DIR}/x64")
    endif()
    if("${TIKI_CUDNN_BIN_DIR}" STREQUAL "")
      set(TIKI_CUDNN_BIN_DIR "${CUDNN_BIN_DIR}")
    endif()
  endif()

  function(tiki_add_dir_definition name)
    if(NOT "${${name}}" STREQUAL "")
      target_compile_definitions(tiki_dirs PRIVATE ${name}="${${name}}")
    endif()
  endfunction()

  # Resolve paths of unfound DLL at runtime.
  if(BUILD_SHARED_LIBS)
    target_link_libraries(tiki PRIVATE "delayimp.lib")
    target_sources(tiki PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/delayload.cpp)
  else()
    # For static library the delayload must be compiled into final executables.
    target_link_libraries(tiki PUBLIC "delayimp.lib")
    target_sources(
      tiki PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/delayload.cpp>)
  endif()
  # Get all the CUDA DLLs we could link with.
  file(
    GLOB CUDA_DLL_NAMES
    RELATIVE "${CUDAToolkit_BIN_DIR}/x64"
    "${CUDAToolkit_BIN_DIR}/x64/*.dll")
  # Delay load CUDA and cuDNN libs.
  foreach(CUDA_DLL ${CUDA_DLL_NAMES} ${CUDNN_DLL_NAMES})
    target_link_options(tiki PUBLIC "/DELAYLOAD:${CUDA_DLL}")
  endforeach()
  # Pass the locations where CUDA DLLs are placed.
  foreach(dir_var TIKI_CUDA_BIN_DIR TIKI_CUDNN_BIN_DIR)
    tiki_add_dir_definition(${dir_var})
  endforeach()
else()
  # For POSIX we rely on RPATH to search for CUDA libs.
  if(TIKI_LOAD_CUDA_LIBS_FROM_PYTHON)
    set_property(
      TARGET tiki
      APPEND
      PROPERTY INSTALL_RPATH
               # The paths here should match the install_requires in setup.py.
               "$ORIGIN/../../nvidia/cu13/lib"
               "$ORIGIN/../../nvidia/cublas/lib"
               "$ORIGIN/../../nvidia/cuda_nvrtc/lib"
               "$ORIGIN/../../nvidia/cudnn/lib"
               "$ORIGIN/../../nvidia/cufft/lib"
               "$ORIGIN/../../nvidia/cusolver/lib"
               "$ORIGIN/../../nvidia/nccl/lib")
  endif()
endif()

# ------------------------ Dependencies ------------------------

# Use fixed version of CCCL.
FetchContent_Declare(
  cccl
  URL "https://github.com/NVIDIA/cccl/releases/download/v3.1.3/cccl-v3.1.3.zip")
FetchContent_MakeAvailable(cccl)
target_include_directories(tiki BEFORE PRIVATE "${cccl_SOURCE_DIR}/include")

# Install CCCL headers for JIT.
install(DIRECTORY ${cccl_SOURCE_DIR}/include/cuda
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cccl)
install(DIRECTORY ${cccl_SOURCE_DIR}/include/nv
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cccl)

# The binary of C++ tests will not be installed so it can not find the CCCL
# headers, and we have to hard-code the path.
if(TIKI_BUILD_TESTS)
  target_compile_definitions(tiki_dirs
                             PRIVATE TIKI_CCCL_DIR="${cccl_SOURCE_DIR}/include")
endif()

# Use fixed version of NVTX.
FetchContent_Declare(
  nvtx3
  GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
  GIT_TAG v3.1.1
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR c EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(nvtx3)
target_link_libraries(tiki PUBLIC $<BUILD_INTERFACE:nvtx3-cpp>)

# Make cuda runtime APIs available in non-cuda files.
target_include_directories(tiki PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Use cublasLt.
target_link_libraries(tiki PRIVATE CUDA::cublasLt)

# Use cuFFT.
target_link_libraries(tiki PRIVATE CUDA::cufft)

# Use cuSOLVER.
target_link_libraries(tiki PRIVATE CUDA::cusolver)

# Use NVRTC and driver APIs.
target_link_libraries(tiki PRIVATE CUDA::nvrtc CUDA::cuda_driver)

# Use the frontend APIs of cuDNN.
FetchContent_Declare(
  cudnn
  GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
  GIT_TAG v1.16.0
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL)
set(CUDNN_FRONTEND_SKIP_JSON_LIB ON)
set(CUDNN_FRONTEND_BUILD_SAMPLES OFF)
set(CUDNN_FRONTEND_BUILD_TESTS OFF)
set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF)
FetchContent_MakeAvailable(cudnn)
target_link_libraries(tiki PRIVATE $<BUILD_INTERFACE:cudnn_frontend>)
# Link with the actual cuDNN libraries.
target_link_libraries(tiki PRIVATE CUDNN::cudnn_all)

# Use header-only CUTLASS.
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG v4.4.2
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR include EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(cutlass)
target_include_directories(
  tiki SYSTEM PRIVATE $<BUILD_INTERFACE:${cutlass_SOURCE_DIR}/include>)

# Install CUTLASS headers for JIT.
install(DIRECTORY ${cutlass_SOURCE_DIR}/include/cute
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${cutlass_SOURCE_DIR}/include/cutlass
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
