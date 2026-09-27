// Copyright © 2024 Apple Inc.
#pragma once

#include <optional>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "tiki/array.h"

namespace tk = tiki::core;
namespace nb = nanobind;

namespace nanobind {

namespace detail {

template <>
struct dtype_traits<tk::float16_t> {
  static constexpr dlpack::dtype value{
      /* code */ uint8_t(nb::dlpack::dtype_code::Float),
      /* bits */ 16,
      /* lanes */ 1};
  static constexpr const char* name = "float16";
};

template <>
struct dtype_traits<tk::bfloat16_t> {
  static constexpr dlpack::dtype value{
      /* code */ uint8_t(nb::dlpack::dtype_code::Bfloat),
      /* bits */ 16,
      /* lanes */ 1};
  static constexpr const char* name = "bfloat16";
};

} // namespace detail

} // namespace nanobind

struct ArrayLike {
  ArrayLike(nb::object obj) : obj(obj) {};
  nb::object obj;
};

tk::array nd_array_to_tiki(
    nb::ndarray<nb::ro> nd_array,
    std::optional<tk::Dtype> mx_dtype,
    std::optional<nb::dlpack::dtype> src_dlpack_dtype_override = std::nullopt,
    std::optional<bool> copy = std::nullopt);

nb::ndarray<nb::numpy> tiki_to_np_array(const tk::array& a);
nb::ndarray<> tiki_to_dlpack(
    const tk::array& a,
    bool force_copy,
    std::optional<std::tuple<int, int>> dl_device);

nb::object to_scalar(tk::array& a);

nb::object tolist(tk::array& a);

tk::array create_array(
    nb::object v,
    std::optional<tk::Dtype> t,
    std::optional<bool> copy = true);
tk::array array_from_list(nb::list pl, std::optional<tk::Dtype> dtype);
tk::array array_from_list(nb::tuple pl, std::optional<tk::Dtype> dtype);

// Narrow a Python-side shape dimension (int64) to a C++ tk::ShapeElem (int32),
// raising a clear error if the value would overflow.
int check_shape_dim(int64_t dim);
