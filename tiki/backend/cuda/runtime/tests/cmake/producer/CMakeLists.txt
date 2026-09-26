cmake_minimum_required(VERSION 3.24)
project(runtime_install C)
include(GNUInstallDirs)

add_library(tiki STATIC tiki.c)
include(../../../../runtime.cmake)

# Use a host archive to test the production install rules without CUDA.
file(MAKE_DIRECTORY ${TIKI_CARGO_TARGET_DIR}/release)
execute_process(
  COMMAND ${CMAKE_C_COMPILER} -c ${CMAKE_CURRENT_SOURCE_DIR}/runtime.c -o
          ${CMAKE_CURRENT_BINARY_DIR}/runtime.o COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND ${CMAKE_AR} qc ${TIKI_CUDA_RUNTIME_LIB}
          ${CMAKE_CURRENT_BINARY_DIR}/runtime.o COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${CMAKE_RANLIB} ${TIKI_CUDA_RUNTIME_LIB}
                        COMMAND_ERROR_IS_FATAL ANY)

install(
  TARGETS tiki
  EXPORT TIKITargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT TIKITargets DESTINATION lib/cmake/Tiki)
