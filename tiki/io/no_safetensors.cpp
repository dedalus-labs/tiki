// Copyright © 2023-2024 Apple Inc.

#include "tiki/io.h"

namespace tiki::core {

SafetensorsLoad load_safetensors(std::shared_ptr<io::Reader>, StreamOrDevice) {
  throw std::runtime_error(
      "[load_safetensors] Compile with TIKI_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

SafetensorsLoad load_safetensors(const std::string&, StreamOrDevice) {
  throw std::runtime_error(
      "[load_safetensors] Compile with TIKI_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

void save_safetensors(
    std::shared_ptr<io::Writer>,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>) {
  throw std::runtime_error(
      "[save_safetensors] Compile with TIKI_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>) {
  throw std::runtime_error(
      "[save_safetensors] Compile with TIKI_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

} // namespace tiki::core
