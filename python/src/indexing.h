// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>

#include "tiki/array.h"
#include "python/src/utils.h"

namespace tk = tiki::core;
namespace nb = nanobind;

tk::array tiki_get_item(const tk::array& src, const nb::object& obj);
void tiki_set_item(
    tk::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
tk::array tiki_add_item(
    const tk::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
tk::array tiki_subtract_item(
    const tk::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
tk::array tiki_multiply_item(
    const tk::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
tk::array tiki_divide_item(
    const tk::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
tk::array tiki_maximum_item(
    const tk::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
tk::array tiki_minimum_item(
    const tk::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
