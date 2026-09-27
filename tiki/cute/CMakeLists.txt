find_program(CARGO cargo REQUIRED)
set(TIKI_CUTE_CARGO_TARGET ${CMAKE_BINARY_DIR}/cute-cargo)
set(TIKI_CUTE_CARGO_ENV CARGO_TARGET_DIR=${TIKI_CUTE_CARGO_TARGET})
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_DEPLOYMENT_TARGET)
  list(APPEND TIKI_CUTE_CARGO_ENV
       MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
endif()
set(TIKI_CUTE_LIB
    ${TIKI_CUTE_CARGO_TARGET}/release/${CMAKE_STATIC_LIBRARY_PREFIX}tiki_cute${CMAKE_STATIC_LIBRARY_SUFFIX}
)
file(GLOB_RECURSE TIKI_CUTE_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs)
add_custom_command(
  OUTPUT ${TIKI_CUTE_LIB}
  COMMAND
    ${CMAKE_COMMAND} -E env ${TIKI_CUTE_CARGO_ENV} ${CARGO} build --locked
    --release --features cxx-bridge --manifest-path
    ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
  DEPENDS ${TIKI_CUTE_SOURCES} Cargo.toml Cargo.lock build.rs
  COMMENT "Building the Rust CuTe layout crate"
  VERBATIM)
add_custom_target(tiki_cute_rust DEPENDS ${TIKI_CUTE_LIB})

nanobind_add_module(tiki_cute_python NB_STATIC STABLE_ABI FREE_THREADED
                    ${PROJECT_SOURCE_DIR}/python/src/layout.cpp)
add_dependencies(tiki_cute_python tiki_cute_rust)
target_include_directories(tiki_cute_python
                           PRIVATE ${TIKI_CUTE_CARGO_TARGET}/cxxbridge)
target_link_libraries(tiki_cute_python PRIVATE ${TIKI_CUTE_LIB}
                                               ${CMAKE_DL_LIBS})
set_target_properties(
  tiki_cute_python
  PROPERTIES OUTPUT_NAME _layout
             LIBRARY_OUTPUT_DIRECTORY
             ${TIKI_PYTHON_BINDINGS_OUTPUT_DIRECTORY}/layout)
install(TARGETS tiki_cute_python LIBRARY DESTINATION layout)
