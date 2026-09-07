// Copyright © 2023-2024 Apple Inc.
#pragma once

#include "tiki/io.h"
#include "tiki/primitives.h"
#include "tiki/transforms.h"
#include "tiki/utils.h"

extern "C" {
#include <gguflib.h>
}

namespace tiki::core {

Shape get_shape(const gguf_tensor& tensor);
void gguf_load_quantized(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor);

} // namespace tiki::core
