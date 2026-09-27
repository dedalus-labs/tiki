# Cargo builds the tiki.layout._cute extension from the PyO3 crate in python/,
# which links tiki-cute. CMake copies it next to the tiki.layout package under
# Python's extension suffix.
find_program(CARGO cargo REQUIRED)
set(TIKI_CUTE_CARGO_TARGET ${CMAKE_BINARY_DIR}/cute-cargo)
set(TIKI_CUTE_CARGO_ENV CARGO_TARGET_DIR=${TIKI_CUTE_CARGO_TARGET}
                        PYO3_PYTHON=${Python_EXECUTABLE})
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_DEPLOYMENT_TARGET)
  list(APPEND TIKI_CUTE_CARGO_ENV
       MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
endif()
set(TIKI_CUTE_CDYLIB
    ${TIKI_CUTE_CARGO_TARGET}/release/${CMAKE_SHARED_LIBRARY_PREFIX}tiki_cute_python${CMAKE_SHARED_LIBRARY_SUFFIX}
)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE TIKI_CUTE_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(TIKI_CUTE_EXTENSION
    ${TIKI_PYTHON_BINDINGS_OUTPUT_DIRECTORY}/layout/_cute${TIKI_CUTE_EXT_SUFFIX}
)
file(GLOB_RECURSE TIKI_CUTE_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs
     ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.rs)
add_custom_command(
  OUTPUT ${TIKI_CUTE_EXTENSION}
  COMMAND
    ${CMAKE_COMMAND} -E env ${TIKI_CUTE_CARGO_ENV} ${CARGO} build --locked
    --release --package tiki-cute-python --features extension-module
    --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
  COMMAND ${CMAKE_COMMAND} -E copy ${TIKI_CUTE_CDYLIB} ${TIKI_CUTE_EXTENSION}
  DEPENDS ${TIKI_CUTE_SOURCES} Cargo.toml Cargo.lock python/Cargo.toml
          python/build.rs
  COMMENT "Building the tiki.layout._cute extension"
  VERBATIM)
add_custom_target(tiki_cute_python ALL DEPENDS ${TIKI_CUTE_EXTENSION})
install(PROGRAMS ${TIKI_CUTE_EXTENSION} DESTINATION layout)
