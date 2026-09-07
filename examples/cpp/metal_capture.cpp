// Copyright © 2024 Apple Inc.

#include <cassert>
#include <iostream>

#include "tiki/tiki.h"

namespace tk = tiki::core;

int main() {
  // To use Metal debugging and profiling:
  // 1. Build with the TIKI_METAL_DEBUG CMake option (i.e. -DTIKI_METAL_DEBUG=ON).
  // 2. Run with MTL_CAPTURE_ENABLED=1.
  tk::metal::start_capture("tiki_trace.gputrace");

  // Start at index two because the default GPU and CPU streams have indices
  // zero and one, respectively. This naming matches the label assigned to each
  // stream's command queue.
  auto s2 = new_stream(tk::Device::gpu);
  auto s3 = new_stream(tk::Device::gpu);

  auto a = tk::arange(1.f, 10.f, 1.f, tk::float32, s2);
  auto b = tk::arange(1.f, 10.f, 1.f, tk::float32, s3);
  auto x = tk::add(a, a, s2);
  auto y = tk::add(b, b, s3);

  // The multiply will happen on the default stream.
  std::cout << tk::multiply(x, y) << std::endl;

  tk::metal::stop_capture();
}
