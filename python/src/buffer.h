// Copyright © 2024 Apple Inc.
#pragma once
#include <optional>

#include <nanobind/nanobind.h>

#include "tiki/array.h"
#include "tiki/utils.h"

// Only defined in >= Python 3.9
// https://github.com/python/cpython/blob/f6cdc6b4a191b75027de342aa8b5d344fb31313e/Include/typeslots.h#L2-L3
#ifndef Py_bf_getbuffer
#define Py_bf_getbuffer 1
#define Py_bf_releasebuffer 2
#endif

namespace tk = tiki::core;
namespace nb = nanobind;

std::string buffer_format(const tk::array& a) {
  // https://docs.python.org/3.10/library/struct.html#format-characters
  switch (a.dtype()) {
    case tk::bool_:
      return "?";
    case tk::uint8:
      return "B";
    case tk::uint16:
      return "H";
    case tk::uint32:
      return "I";
    case tk::uint64:
      return "Q";
    case tk::int8:
      return "b";
    case tk::int16:
      return "h";
    case tk::int32:
      return "i";
    case tk::int64:
      return "q";
    case tk::float16:
      return "e";
    case tk::float32:
      return "f";
    case tk::bfloat16:
      return "bfloat16";
    case tk::float64:
      return "d";
    case tk::complex64:
      return "Zf\0";
    default: {
      std::ostringstream os;
      os << "bad dtype: " << a.dtype();
      throw std::runtime_error(os.str());
    }
  }
}

struct buffer_info {
  std::string format;
  std::vector<Py_ssize_t> shape;
  std::vector<Py_ssize_t> strides;

  buffer_info(
      std::string format,
      std::vector<Py_ssize_t> shape_in,
      std::vector<Py_ssize_t> strides_in)
      : format(std::move(format)),
        shape(std::move(shape_in)),
        strides(std::move(strides_in)) {}

  buffer_info(const buffer_info&) = delete;
  buffer_info& operator=(const buffer_info&) = delete;

  buffer_info(buffer_info&& other) noexcept {
    (*this) = std::move(other);
  }

  buffer_info& operator=(buffer_info&& rhs) noexcept {
    format = std::move(rhs.format);
    shape = std::move(rhs.shape);
    strides = std::move(rhs.strides);
    return *this;
  }
};

extern "C" inline int getbuffer(PyObject* obj, Py_buffer* view, int flags) {
  std::memset(view, 0, sizeof(Py_buffer));
  auto a = nb::cast<tk::array>(nb::handle(obj));

  {
    nb::gil_scoped_release nogil;
    a.eval();
  }

  std::vector<Py_ssize_t> shape(a.shape().begin(), a.shape().end());
  std::vector<Py_ssize_t> strides(a.strides().begin(), a.strides().end());
  for (auto& s : strides) {
    s *= a.itemsize();
  }
  buffer_info* info =
      new buffer_info(buffer_format(a), std::move(shape), std::move(strides));

  view->obj = obj;
  view->ndim = a.ndim();
  view->internal = info;
  view->buf = a.data<void>();
  view->itemsize = a.itemsize();
  view->len = a.nbytes();
  view->readonly = false;
  if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
    view->format = const_cast<char*>(info->format.c_str());
  }
  if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
    view->strides = info->strides.data();
    view->shape = info->shape.data();
  }
  Py_INCREF(view->obj);
  return 0;
}

extern "C" inline void releasebuffer(PyObject*, Py_buffer* view) {
  delete (buffer_info*)view->internal;
}
