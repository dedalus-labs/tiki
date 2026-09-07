// Copyright © 2026 Dedalus Labs, Inc.

#include <stdexcept>

#include <nanobind/nanobind.h>

#include "rust/cxx.h"
#include "tiki-layout/src/bridge.rs.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

class LayoutError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

template <typename Function>
auto checked(Function&& function) {
  try {
    return function();
  } catch (const rust::Error& error) {
    throw LayoutError(error.what());
  }
}

struct Swizzle {
  Swizzle(int64_t bits, int64_t base, int64_t shift)
      : value(checked([&] {
          return mlx::core::layout_rt::new_swizzle(bits, base, shift);
        })) {}

  rust::Box<mlx::core::layout_rt::Swizzle> value;
};

} // namespace

NB_MODULE(_layout, m) {
  nb::exception<LayoutError>(m, "LayoutError", PyExc_ValueError);
  nb::class_<Swizzle>(m, "Swizzle")
      .def(nb::init<int64_t, int64_t, int64_t>(), "bits"_a, "base"_a, "shift"_a)
      .def_prop_ro("bits", [](const Swizzle& s) { return s.value->bits(); })
      .def_prop_ro("base", [](const Swizzle& s) { return s.value->base(); })
      .def_prop_ro("shift", [](const Swizzle& s) { return s.value->shift(); })
      .def(
          "__call__",
          [](const Swizzle& s, int64_t index) {
            return checked([&] { return s.value->apply(index); });
          },
          "index"_a.noconvert())
      .def(
          "__eq__",
          [](const Swizzle& a, const Swizzle& b) {
            return a.value->bits() == b.value->bits() &&
                a.value->base() == b.value->base() &&
                a.value->shift() == b.value->shift();
          },
          nb::is_operator())
      .def("__hash__", [](const Swizzle& s) {
        return nb::hash(
            nb::make_tuple(s.value->bits(), s.value->base(), s.value->shift()));
      });
}
