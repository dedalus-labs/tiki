// Copyright © 2023-2024 Apple Inc.
#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "python/src/small_vector.h"
#include "tiki/backend/metal/metal.h"
#include "tiki/device.h"
#include "tiki/memory.h"

namespace tk = tiki::core;
namespace nb = nanobind;
using namespace nb::literals;

bool DEPRECATE(const char* old_fn, const char* new_fn) {
  std::cerr << old_fn << " is deprecated and will be removed in a future "
            << "version. Use " << new_fn << " instead." << std::endl;
  return true;
}

#define DEPRECATE(oldfn, newfn) static bool dep = DEPRECATE(oldfn, newfn)

void init_metal(nb::module_& m) {
  nb::module_ metal = m.def_submodule("metal", "tiki.metal");
  metal.def(
      "is_available",
      &tk::metal::is_available,
      R"pbdoc(
      Check if the Metal back-end is available.
      )pbdoc");
  metal.def("get_active_memory", []() {
    DEPRECATE("tk.metal.get_active_memory", "tk.get_active_memory");
    return tk::get_active_memory();
  });
  metal.def("get_peak_memory", []() {
    DEPRECATE("tk.metal.get_peak_memory", "tk.get_peak_memory");
    return tk::get_peak_memory();
  });
  metal.def("reset_peak_memory", []() {
    DEPRECATE("tk.metal.reset_peak_memory", "tk.reset_peak_memory");
    tk::reset_peak_memory();
  });
  metal.def("get_cache_memory", []() {
    DEPRECATE("tk.metal.get_cache_memory", "tk.get_cache_memory");
    return tk::get_cache_memory();
  });
  metal.def(
      "set_memory_limit",
      [](size_t limit) {
        DEPRECATE("tk.metal.set_memory_limit", "tk.set_memory_limit");
        return tk::set_memory_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_cache_limit",
      [](size_t limit) {
        DEPRECATE("tk.metal.set_cache_limit", "tk.set_cache_limit");
        return tk::set_cache_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_wired_limit",
      [](size_t limit) {
        DEPRECATE("tk.metal.set_wired_limit", "tk.set_wired_limit");
        return tk::set_wired_limit(limit);
      },
      "limit"_a);
  metal.def("clear_cache", []() {
    DEPRECATE("tk.metal.clear_cache", "tk.clear_cache");
    tk::clear_cache();
  });
  metal.def(
      "start_capture",
      &tk::metal::start_capture,
      "path"_a,
      R"pbdoc(
      Start a Metal capture.

      Args:
        path (str): The path to save the capture which should have
          the extension ``.gputrace``.
      )pbdoc");
  metal.def(
      "stop_capture",
      &tk::metal::stop_capture,
      R"pbdoc(
      Stop a Metal capture.
      )pbdoc");
  metal.def("device_info", []() {
    DEPRECATE("tk.metal.device_info", "tk.device_info");
    return tk::device_info(tk::Device(tk::Device::gpu, 0));
  });
}
