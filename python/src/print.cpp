#include <cstdint>
#include <cstring>
#include <sstream>

#include <nanobind/typing.h>

#include "python/src/utils.h"
#include "tiki/utils.h"

#include "tiki/tiki.h"

namespace tk = tiki::core;
namespace nb = nanobind;
using namespace nb::literals;

struct PrintOptionsContext {
  tk::PrintOptions old_options;
  tk::PrintOptions new_options;
  PrintOptionsContext(tk::PrintOptions p) : new_options(p) {}
  PrintOptionsContext& enter() {
    old_options = tk::get_global_formatter().format_options;
    tk::set_printoptions(new_options);
    return *this;
  }
  void exit(nb::args) {
    tk::set_printoptions(old_options);
  }
};

void init_print(nb::module_& m) {
  // Set Python print formatting options
  tk::get_global_formatter().capitalize_bool = true;
  // Expose printing options to Python: allow setting global precision.
  nb::class_<tk::PrintOptions>(m, "PrintOptions")
      .def(nb::init<int>(), "precision"_a = -1)
      .def_rw("precision", &tk::PrintOptions::precision);

  m.def(
      "set_printoptions",
      [](int precision) { tk::set_printoptions({precision}); },
      "precision"_a = tk::get_global_formatter().format_options.precision,
      R"pbdoc(
        Set global printing precision for array formatting.

        Example:
            >>> print(x)  # Uses default precision
            >>> tk.set_printoptions(precision=3)
            >>> print(x)  # Uses precision of 3
            >>> print(x)  # Uses precision of 3 (again)

        Args:
            precision (int): Number of decimal places.
        )pbdoc");
  m.def(
      "get_printoptions",
      []() { return tk::get_global_formatter().format_options; },
      R"pbdoc(
        Get global printing precision for array formatting.

        Returns:
        PrintOptions: The format options used for printing arrays.
        )pbdoc");

  nb::class_<PrintOptionsContext>(m, "_PrintOptionsContext")
      .def(nb::init<tk::PrintOptions>())
      .def("__enter__", &PrintOptionsContext::enter)
      .def("__exit__", &PrintOptionsContext::exit);

  m.def(
      "printoptions",
      [](int precision) { return PrintOptionsContext({precision}); },
      "precision"_a = tk::get_global_formatter().format_options.precision,
      R"pbdoc(
        Context manager for setting print options temporarily.

        Example:
            >>> print(x)  # Uses default precision
            >>> with tk.printoptions(precision=3):
            >>>     print(x)  # Uses precision of 3
            >>> print(x)  # Back to default precision


        Args:
            precision (int): Number of decimal places. Use -1 for default
        )pbdoc");
}
