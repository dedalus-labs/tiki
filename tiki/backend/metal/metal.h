// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "tiki/api.h"

namespace tiki::core::metal {

/* Check if the Metal backend is available. */
TIKI_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
TIKI_API void start_capture(std::string path = "");
TIKI_API void stop_capture();

/** Get information about the GPU and system settings. */
TIKI_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/* Set a custom path to tiki.metallib. Must be called before any Tiki operation.
 */
TIKI_API void set_metallib_path(const std::string& path);
TIKI_API const std::string& get_metallib_path();

} // namespace tiki::core::metal
