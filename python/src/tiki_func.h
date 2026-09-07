// Copyright © 2025 Apple Inc.

#pragma once

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>

namespace nb = nanobind;
using namespace nb::literals;

nb::callable tiki_func(
    nb::object func,
    const nb::callable& orig_func,
    std::vector<PyObject*> deps);

template <typename F, typename... Deps>
nb::callable tiki_func(F func, const nb::callable& orig_func, Deps&&... deps) {
  return tiki_func(
      nb::cpp_function(std::move(func)),
      orig_func,
      std::vector<PyObject*>{deps.ptr()...});
}

template <typename... Deps>
nb::callable
tiki_func(nb::object func, const nb::callable& orig_func, Deps&&... deps) {
  return tiki_func(
      std::move(func), orig_func, std::vector<PyObject*>{deps.ptr()...});
}
