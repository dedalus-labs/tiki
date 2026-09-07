// Copyright © 2024 Apple Inc.

#include "python/src/utils.h"
#include "python/src/convert.h"
#include "tiki/ops.h"
#include "tiki/utils.h"

tk::array to_array(
    const ScalarOrArray& v,
    std::optional<tk::Dtype> dtype /* = std::nullopt */) {
  if (auto pv = std::get_if<nb::bool_>(&v); pv) {
    return tk::array(nb::cast<bool>(*pv), dtype.value_or(tk::bool_));
  } else if (auto pv = std::get_if<nb::int_>(&v); pv) {
    auto val = nb::cast<int64_t>(*pv);
    auto default_type = (val > std::numeric_limits<int>::max() ||
                         val < std::numeric_limits<int>::min())
        ? tk::int64
        : tk::int32;
    auto out_t = dtype.value_or(default_type);
    if (tk::issubdtype(out_t, tk::integer) && out_t.size() < 8) {
      auto info = tk::iinfo(out_t);
      if (val < info.min || val > static_cast<int64_t>(info.max)) {
        std::ostringstream msg;
        msg << "Converting " << val << " to " << out_t
            << " would result in overflow.";
        throw std::invalid_argument(msg.str());
      }
    }

    // bool_ is an exception and is always promoted
    return tk::array(val, (out_t == tk::bool_) ? tk::int32 : out_t);
  } else if (auto pv = std::get_if<nb::float_>(&v); pv) {
    auto out_t = dtype.value_or(tk::float32);
    if (!tk::issubdtype(out_t, tk::floating)) {
      out_t = tk::float32;
    }
    if (out_t == tk::float64) {
      // Cast straight to double: going through float would round the value
      // to float32 precision while the result still advertises float64.
      return tk::array(nb::cast<double>(*pv), out_t);
    }
    return tk::array(nb::cast<float>(*pv), out_t);
  } else if (auto pv = std::get_if<std::complex<float>>(&v); pv) {
    return tk::array(static_cast<tk::complex64_t>(*pv), tk::complex64);
  } else if (auto pv = std::get_if<tk::array>(&v); pv) {
    return *pv;
  } else if (auto pv = std::get_if<nb::ndarray<nb::ro>>(&v); pv) {
    return nd_array_to_tiki(*pv, dtype);
  } else {
    return to_array_with_accessor(std::get<ArrayLike>(v).obj);
  }
}

std::pair<tk::array, tk::array> to_arrays(
    const ScalarOrArray& a,
    const ScalarOrArray& b) {
  // Four cases:
  // - If both a and b are arrays leave their types alone
  // - If a is an array but b is not, treat b as a weak python type
  // - If b is an array but a is not, treat a as a weak python type
  // - If neither is an array convert to arrays but leave their types alone
  auto is_tiki_array = [](const ScalarOrArray& x) {
    return std::holds_alternative<tk::array>(x) ||
        std::holds_alternative<ArrayLike>(x) &&
        nb::hasattr(std::get<ArrayLike>(x).obj, "__tiki_array__");
  };
  auto get_tiki_array = [](const ScalarOrArray& x) {
    if (auto px = std::get_if<tk::array>(&x); px) {
      return *px;
    } else {
      return nb::cast<tk::array>(
          std::get<ArrayLike>(x).obj.attr("__tiki_array__"));
    }
  };

  if (is_tiki_array(a)) {
    auto arr_a = get_tiki_array(a);
    if (is_tiki_array(b)) {
      auto arr_b = get_tiki_array(b);
      return {arr_a, arr_b};
    }
    return {arr_a, to_array(b, arr_a.dtype())};
  } else if (is_tiki_array(b)) {
    auto arr_b = get_tiki_array(b);
    return {to_array(a, arr_b.dtype()), arr_b};
  } else {
    return {to_array(a), to_array(b)};
  }
}

tk::array to_array_with_accessor(nb::object obj) {
  if (nb::isinstance<tk::array>(obj)) {
    return nb::cast<tk::array>(obj);
  } else if (nb::hasattr(obj, "__tiki_array__")) {
    return nb::cast<tk::array>(obj.attr("__tiki_array__")());
  } else {
    std::ostringstream msg;
    msg << "Invalid type " << nb::type_name(obj.type()).c_str()
        << " received in array initialization.";
    throw std::invalid_argument(msg.str());
  }
}
