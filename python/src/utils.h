// Copyright © 2023-2024 Apple Inc.
#pragma once
#include <numeric>
#include <optional>
#include <string>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/variant.h>

#include "tiki/array.h"
#include "python/src/convert.h"

namespace tk = tiki::core;
namespace nb = nanobind;

using IntOrVec = std::variant<std::monostate, int, std::vector<int>>;
using ScalarOrArray = std::variant<
    nb::bool_,
    nb::int_,
    nb::float_,
    // Must be above ndarray
    tk::array,
    // Must be above complex
    nb::ndarray<nb::ro>,
    std::complex<float>,
    ArrayLike>;

inline std::vector<int> get_reduce_axes(const IntOrVec& v, int dims) {
  std::vector<int> axes;
  if (std::holds_alternative<std::monostate>(v)) {
    axes.resize(dims);
    std::iota(axes.begin(), axes.end(), 0);
  } else if (auto pv = std::get_if<int>(&v); pv) {
    axes.push_back(*pv);
  } else {
    axes = std::get<std::vector<int>>(v);
  }
  return axes;
}

inline bool is_comparable_with_array(const ScalarOrArray& v) {
  // Checks if the value can be compared to an array (or is already an
  // tiki array)
  if (auto pv = std::get_if<ArrayLike>(&v); pv) {
    auto obj = (*pv).obj;
    return nb::isinstance<tk::array>(obj) || nb::hasattr(obj, "__tiki_array__");
  } else {
    // If it's not an object, it's a scalar (nb::int_, nb::float_, etc.)
    // and can be compared to an array
    return true;
  }
}

inline nb::handle get_handle_of_object(const ScalarOrArray& v) {
  return std::get<ArrayLike>(v).obj.ptr();
}

inline void throw_invalid_operation(
    const std::string& operation,
    const ScalarOrArray operand) {
  std::ostringstream msg;
  msg << "Cannot perform " << operation << " on an tiki.array and "
      << nb::type_name(get_handle_of_object(operand).type()).c_str();
  throw std::invalid_argument(msg.str());
}

tk::array to_array(
    const ScalarOrArray& v,
    std::optional<tk::Dtype> dtype = std::nullopt);

std::pair<tk::array, tk::array> to_arrays(
    const ScalarOrArray& a,
    const ScalarOrArray& b);

tk::array to_array_with_accessor(nb::object obj);
