// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include "tiki/io.h"

namespace tk = tiki::core;
namespace nb = nanobind;

using LoadOutputTypes = std::variant<
    tk::array,
    std::unordered_map<std::string, tk::array>,
    tk::SafetensorsLoad,
    tk::GGUFLoad>;

tk::SafetensorsLoad tiki_load_safetensor_helper(
    nb::object file,
    tk::StreamOrDevice s);
void tiki_save_safetensor_helper(
    nb::object file,
    nb::dict d,
    std::optional<nb::dict> m);

tk::GGUFLoad tiki_load_gguf_helper(nb::object file, tk::StreamOrDevice s);

void tiki_save_gguf_helper(
    nb::object file,
    nb::dict d,
    std::optional<nb::dict> m);

LoadOutputTypes tiki_load_helper(
    nb::object file,
    std::optional<std::string> format,
    bool return_metadata,
    tk::StreamOrDevice s);
void tiki_save_helper(nb::object file, tk::array a);
void tiki_savez_helper(
    nb::object file,
    nb::args args,
    const nb::kwargs& kwargs,
    bool compressed = false);
