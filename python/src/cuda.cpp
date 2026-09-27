// Copyright © 2023-2025 Apple Inc.

#include <nanobind/nanobind.h>

#include "tiki/backend/cuda/cuda.h"

namespace tk = tiki::core;
namespace nb = nanobind;

void init_cuda(nb::module_& m) {
  nb::module_ cuda = m.def_submodule("cuda", "tiki.cuda");

  cuda.def(
      "is_available",
      &tk::cu::is_available,
      R"pbdoc(
      Check if the CUDA back-end is available.
      )pbdoc");
}
