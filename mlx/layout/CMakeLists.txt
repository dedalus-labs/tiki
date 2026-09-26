find_program(CARGO cargo REQUIRED)
set(MLX_LAYOUT_CARGO_TARGET ${CMAKE_BINARY_DIR}/layout-cargo)
set(MLX_LAYOUT_CARGO_ENV CARGO_TARGET_DIR=${MLX_LAYOUT_CARGO_TARGET})
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_DEPLOYMENT_TARGET)
  list(APPEND MLX_LAYOUT_CARGO_ENV
       MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
endif()
set(MLX_LAYOUT_LIB
    ${MLX_LAYOUT_CARGO_TARGET}/release/${CMAKE_STATIC_LIBRARY_PREFIX}tiki_layout${CMAKE_STATIC_LIBRARY_SUFFIX}
)
file(GLOB MLX_LAYOUT_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs)
add_custom_command(
  OUTPUT ${MLX_LAYOUT_LIB}
  COMMAND
    ${CMAKE_COMMAND} -E env ${MLX_LAYOUT_CARGO_ENV} ${CARGO} build --locked
    --release --features cxx-bridge --manifest-path
    ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
  DEPENDS ${MLX_LAYOUT_SOURCES} Cargo.toml Cargo.lock build.rs
  COMMENT "Building Rust indexing transforms"
  VERBATIM)
add_custom_target(tiki_layout_rust DEPENDS ${MLX_LAYOUT_LIB})

nanobind_add_module(tiki_layout_python NB_STATIC STABLE_ABI FREE_THREADED
                    ${PROJECT_SOURCE_DIR}/python/src/layout.cpp)
add_dependencies(tiki_layout_python tiki_layout_rust)
target_include_directories(tiki_layout_python
                           PRIVATE ${MLX_LAYOUT_CARGO_TARGET}/cxxbridge)
target_link_libraries(tiki_layout_python PRIVATE ${MLX_LAYOUT_LIB}
                                                 ${CMAKE_DL_LIBS})
set_target_properties(
  tiki_layout_python
  PROPERTIES OUTPUT_NAME _layout LIBRARY_OUTPUT_DIRECTORY
                                 ${MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY}/tiki)
install(TARGETS tiki_layout_python LIBRARY DESTINATION tiki)
