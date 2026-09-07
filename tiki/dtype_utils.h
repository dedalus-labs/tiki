// Copyright © 2025 Apple Inc.

#pragma once

#include <sstream>

#include "tiki/dtype.h"
#include "tiki/utils.h"

namespace tiki::core {

// Return string representation of dtype.
const char* dtype_to_string(Dtype arg);

#define TIKI_INTERNAL_DTYPE_SWITCH_CASE(DTYPE, TYPE) \
  case DTYPE:                                       \
    f(type_identity<TYPE>{});                       \
    break

#define TIKI_INTERNAL_DTYPE_SWITCH_INTS()            \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(int8, int8_t);     \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(int16, int16_t);   \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(int32, int32_t);   \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(int64, int64_t);   \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(uint8, uint8_t);   \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(uint16, uint16_t); \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(uint32, uint32_t); \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(uint64, uint64_t)

#define TIKI_INTERNAL_DTYPE_SWITCH_FLOATS()              \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(float16, float16_t);   \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(bfloat16, bfloat16_t); \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(float32, float);       \
  TIKI_INTERNAL_DTYPE_SWITCH_CASE(float64, double)

// This already exists in C++20 but in C++20 we can also just use templated
// lambdas which will make this so much nicer.
template <typename T>
struct type_identity {
  using type = T;
};

#define TIKI_GET_TYPE(x) typename decltype(x)::type
#define TIKI_GET_VALUE(x) decltype(x)::value

template <typename F>
void dispatch_all_types(Dtype dt, F&& f) {
  switch (dt) {
    TIKI_INTERNAL_DTYPE_SWITCH_CASE(bool_, bool);
    TIKI_INTERNAL_DTYPE_SWITCH_INTS();
    TIKI_INTERNAL_DTYPE_SWITCH_FLOATS();
    TIKI_INTERNAL_DTYPE_SWITCH_CASE(complex64, complex64_t);
  }
}

template <typename F>
void dispatch_int_types(Dtype dt, std::string_view tag, F&& f) {
  switch (dt) {
    TIKI_INTERNAL_DTYPE_SWITCH_INTS();
    default:
      std::ostringstream msg;
      msg << tag << " Only integer types supported but " << dt
          << " was provided";
      throw std::invalid_argument(msg.str());
  }
}

template <typename F>
void dispatch_float_types(Dtype dt, std::string_view tag, F&& f) {
  switch (dt) {
    TIKI_INTERNAL_DTYPE_SWITCH_FLOATS();
    default:
      std::ostringstream msg;
      msg << tag << " Only float types supported but " << dt << " was provided";
      throw std::invalid_argument(msg.str());
  }
}

template <typename F>
void dispatch_inexact_types(Dtype dt, std::string_view tag, F&& f) {
  switch (dt) {
    TIKI_INTERNAL_DTYPE_SWITCH_FLOATS();
    TIKI_INTERNAL_DTYPE_SWITCH_CASE(complex64, complex64_t);
    default:
      std::ostringstream msg;
      msg << tag << " Only inexact (float/complex) types supported but " << dt
          << " was provided";
      throw std::invalid_argument(msg.str());
  }
}

template <typename F>
void dispatch_int_float_types(Dtype dt, std::string_view tag, F&& f) {
  switch (dt) {
    TIKI_INTERNAL_DTYPE_SWITCH_INTS();
    TIKI_INTERNAL_DTYPE_SWITCH_FLOATS();
    default:
      std::ostringstream msg;
      msg << tag << " Only integer and float types supported but " << dt
          << " was provided";
      throw std::invalid_argument(msg.str());
  }
}

template <typename F>
void dispatch_real_types(Dtype dt, std::string_view tag, F&& f) {
  switch (dt) {
    TIKI_INTERNAL_DTYPE_SWITCH_CASE(bool_, bool);
    TIKI_INTERNAL_DTYPE_SWITCH_INTS();
    TIKI_INTERNAL_DTYPE_SWITCH_FLOATS();
    default:
      std::ostringstream msg;
      msg << tag << " Only real numbers supported but " << dt
          << " was provided";
      throw std::invalid_argument(msg.str());
  }
}

} // namespace tiki::core
