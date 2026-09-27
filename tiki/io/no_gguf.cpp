// Copyright © 2023-2024 Apple Inc.

#include "tiki/io.h"

namespace tiki::core {

GGUFLoad load_gguf(const std::string&, StreamOrDevice s) {
  throw std::runtime_error(
      "[load_gguf] Compile with TIKI_BUILD_GGUF=ON to enable GGUF support.");
}

void save_gguf(
    std::string,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, GGUFMetaData>) {
  throw std::runtime_error(
      "[save_gguf] Compile with TIKI_BUILD_GGUF=ON to enable GGUF support.");
}

} // namespace tiki::core
