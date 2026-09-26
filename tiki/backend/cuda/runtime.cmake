# Storage ownership lives in the Rust crate under runtime/. Cargo builds it and
# generates the CXX header that allocator.cpp includes.
find_program(CARGO cargo REQUIRED)
include(GNUInstallDirs)
set(TIKI_CUDA_RUNTIME_DIR ${CMAKE_CURRENT_LIST_DIR}/runtime)
set(TIKI_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo)
set(TIKI_CUDA_RUNTIME_LIB
    ${TIKI_CARGO_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}tiki_cuda_runtime${CMAKE_STATIC_LIBRARY_SUFFIX}
)
file(GLOB_RECURSE TIKI_CUDA_RUNTIME_SOURCES ${TIKI_CUDA_RUNTIME_DIR}/src/*.rs
     ${TIKI_CUDA_RUNTIME_DIR}/src/*.cpp)
add_custom_command(
  OUTPUT ${TIKI_CUDA_RUNTIME_LIB}
  COMMAND
    ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${TIKI_CARGO_TARGET_DIR}
    "CUDA_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}" ${CARGO} build --locked
    --release --manifest-path ${TIKI_CUDA_RUNTIME_DIR}/Cargo.toml
  DEPENDS ${TIKI_CUDA_RUNTIME_SOURCES} ${TIKI_CUDA_RUNTIME_DIR}/Cargo.toml
          ${TIKI_CUDA_RUNTIME_DIR}/Cargo.lock ${TIKI_CUDA_RUNTIME_DIR}/build.rs
  COMMENT "Building tiki-cuda-runtime"
  VERBATIM)
add_custom_target(tiki_cuda_runtime DEPENDS ${TIKI_CUDA_RUNTIME_LIB})
add_dependencies(tiki tiki_cuda_runtime)
target_include_directories(tiki PRIVATE ${TIKI_CARGO_TARGET_DIR}/cxxbridge)
set(TIKI_CUDA_RUNTIME_INSTALL_LIB
    ${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}tiki_cuda_runtime${CMAKE_STATIC_LIBRARY_SUFFIX}
)
if(NOT IS_ABSOLUTE "${TIKI_CUDA_RUNTIME_INSTALL_LIB}")
  set(TIKI_CUDA_RUNTIME_INSTALL_LIB
      "$<INSTALL_PREFIX>/${TIKI_CUDA_RUNTIME_INSTALL_LIB}")
endif()
target_link_libraries(
  tiki
  PRIVATE "$<BUILD_INTERFACE:${TIKI_CUDA_RUNTIME_LIB}>"
          "$<INSTALL_INTERFACE:${TIKI_CUDA_RUNTIME_INSTALL_LIB}>"
          ${CMAKE_DL_LIBS})
if(NOT BUILD_SHARED_LIBS)
  install(FILES ${TIKI_CUDA_RUNTIME_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
