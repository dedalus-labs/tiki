// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <limits>
#include <sstream>
#include <tuple>

#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>

#include "python/src/convert.h"
#include "python/src/utils.h"

#include "tiki/allocator.h"
#include "tiki/backend/common/utils.h"
#include "tiki/backend/cuda/cuda.h"
#include "tiki/backend/metal/metal.h"
#include "tiki/dtype_utils.h"
#include "tiki/ops.h"
#include "tiki/utils.h"

// Defined in ops.cpp.
namespace tiki::core {
array astype(array a, Dtype dtype, bool force_copy, StreamOrDevice s = {});
}

enum PyScalarT {
  pybool = 0,
  pyint = 1,
  pyfloat = 2,
  pycomplex = 3,
};

int check_shape_dim(int64_t dim) {
  if (dim > std::numeric_limits<int>::max() ||
      dim < std::numeric_limits<int>::min()) {
    std::ostringstream msg;
    msg << "Shape dimension " << dim << " is outside the supported range ["
        << std::numeric_limits<int>::min() << ", "
        << std::numeric_limits<int>::max()
        << "]. Tiki currently uses 32-bit integers for shape dimensions.";
    PyErr_SetString(PyExc_OverflowError, msg.str().c_str());
    nb::detail::raise_python_error();
  }
  return static_cast<int>(dim);
}

tk::Shape get_shape(const nb::ndarray<nb::ro>& nd_array) {
  tk::Shape shape;
  shape.reserve(nd_array.ndim());
  for (int i = 0; i < nd_array.ndim(); i++) {
    shape.push_back(check_shape_dim(nd_array.shape(i)));
  }
  return shape;
}

tk::Strides get_strides(const nb::ndarray<nb::ro>& nd_array) {
  tk::Strides strides;
  strides.reserve(nd_array.ndim());
  for (int i = 0; i < nd_array.ndim(); i++) {
    strides.push_back(nd_array.stride(i));
  }
  return strides;
}

size_t strided_storage_size(
    const tk::Shape& shape,
    const tk::Strides& strides) {
  size_t storage_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    if (shape[i] == 0) {
      return 0;
    }
    if (strides[i] < 0) {
      throw std::invalid_argument(
          "Cannot convert DLPack arrays with negative strides to tiki array.");
    }
    storage_size += (shape[i] - 1) * strides[i];
  }
  return storage_size;
}

auto get_strided_layout(
    const nb::ndarray<nb::ro>& nd_array,
    const tk::Shape& shape) {
  auto strides = get_strides(nd_array);
  auto storage_size = strided_storage_size(shape, strides);
  auto [no_bsx_size, is_row_contiguous, is_col_contiguous] = shape.empty()
      ? std::make_tuple(storage_size, true, true)
      : tk::check_contiguity(shape, strides);
  tk::array::Flags flags{
      no_bsx_size == storage_size,
      is_row_contiguous,
      is_col_contiguous,
  };
  return std::make_tuple(storage_size, std::move(strides), flags);
}

template <typename F>
auto dispatch_dlpack_dtype(
    nb::dlpack::dtype type,
    F&& f,
    const char* error_message) {
  if (type == nb::dtype<bool>()) {
    return f.template operator()<bool>(tk::bool_);
  } else if (type == nb::dtype<uint8_t>()) {
    return f.template operator()<uint8_t>(tk::uint8);
  } else if (type == nb::dtype<uint16_t>()) {
    return f.template operator()<uint16_t>(tk::uint16);
  } else if (type == nb::dtype<uint32_t>()) {
    return f.template operator()<uint32_t>(tk::uint32);
  } else if (type == nb::dtype<uint64_t>()) {
    return f.template operator()<uint64_t>(tk::uint64);
  } else if (type == nb::dtype<int8_t>()) {
    return f.template operator()<int8_t>(tk::int8);
  } else if (type == nb::dtype<int16_t>()) {
    return f.template operator()<int16_t>(tk::int16);
  } else if (type == nb::dtype<int32_t>()) {
    return f.template operator()<int32_t>(tk::int32);
  } else if (type == nb::dtype<int64_t>()) {
    return f.template operator()<int64_t>(tk::int64);
  } else if (type == nb::dtype<tk::float16_t>()) {
    return f.template operator()<tk::float16_t>(tk::float16);
  } else if (type == nb::dtype<tk::bfloat16_t>()) {
    return f.template operator()<tk::bfloat16_t>(tk::bfloat16);
  } else if (type == nb::dtype<float>()) {
    return f.template operator()<float>(tk::float32);
  } else if (type == nb::dtype<double>()) {
    return f.template operator()<double>(tk::float32);
  } else if (type == nb::dtype<std::complex<float>>()) {
    return f.template operator()<tk::complex64_t>(tk::complex64);
  } else if (type == nb::dtype<std::complex<double>>()) {
    return f.template operator()<tk::complex128_t>(tk::complex64);
  } else {
    throw std::invalid_argument(error_message);
  }
}

tk::Dtype tiki_dtype_from_dlpack(
    nb::dlpack::dtype type,
    const char* error_message) {
  return dispatch_dlpack_dtype(
      type, []<typename T>(tk::Dtype dtype) { return dtype; }, error_message);
}

nb::dlpack::dtype tiki_dtype_to_dl_dtype(tk::Dtype dtype) {
  nb::dlpack::dtype result;
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = TIKI_GET_TYPE(type_tag);
    result = nb::dtype<T>();
  });
  return result;
}

template <typename SrcT>
tk::array cpu_nd_array_to_tiki(
    nb::ndarray<nb::ro> nd_array,
    const tk::Shape& shape,
    tk::Dtype dst_dtype) {
  auto out = tk::array(shape, dst_dtype, nullptr, {});
  auto [storage_size, strides, flags] = get_strided_layout(nd_array, shape);
  out.set_data(
      tk::allocator::malloc(storage_size * tk::size_of(dst_dtype)),
      storage_size,
      std::move(strides),
      flags);
  if (storage_size > 0) {
    dispatch_all_types(
        dst_dtype, [&, storage_size = storage_size](auto type_tag) {
          using DstT = TIKI_GET_TYPE(type_tag);
          auto src = static_cast<const SrcT*>(nd_array.data());
          auto dst = out.data<DstT>();
          std::copy(src, src + storage_size, dst);
        });
  }
  out.set_status(tk::array::Status::available);
  return out;
}

// Try to adopt a CPU host buffer as an tiki array without copying. On unified
// memory the host pointer is GPU-addressable, so we wrap it directly via the
// allocator instead of copying. The bytes are reinterpreted rather than
// converted, so the source element width must already match the destination
// dtype. The source ndarray is kept alive for the lifetime of the returned
// array.
//
// Returns std::nullopt when the buffer cannot be adopted (no Metal backend,
// dtype width mismatch, or a pointer the platform will not wrap), so the caller
// can fall back to a copy or raise.
std::optional<tk::array> cpu_nd_array_to_tiki_no_copy(
    nb::ndarray<nb::ro> nd_array,
    const tk::Shape& shape,
    tk::Dtype dst_dtype) {
  if (!tk::metal::is_available() ||
      nd_array.itemsize() != tk::size_of(dst_dtype)) {
    return std::nullopt;
  }

  auto [storage_size, strides, flags] = get_strided_layout(nd_array, shape);
  auto buf = tk::allocator::make_buffer(
      const_cast<void*>(nd_array.data()),
      storage_size * tk::size_of(dst_dtype));
  // make_buffer returns a null buffer when the pointer cannot be wrapped, e.g.
  // when its alignment is not accepted by the platform.
  if (buf.ptr() == nullptr) {
    return std::nullopt;
  }

  tk::array out(shape, dst_dtype, nullptr, {});
  out.set_data(
      buf,
      storage_size,
      std::move(strides),
      flags,
      nd_array.byte_offset(),
      // The buffer wraps caller-owned memory, so release the wrapper rather
      // than returning it to the allocator's reuse pool, which must only
      // recycle buffers it allocated itself.
      [owner = std::move(nd_array)](tk::allocator::Buffer b) {
        tk::allocator::release(b);
      });
  out.set_status(tk::array::Status::available);
  return out;
}

tk::array metal_nd_array_to_tiki(
    nb::ndarray<nb::ro> nd_array,
    tk::Dtype src_dtype,
    tk::Dtype dst_dtype,
    bool copy) {
  if (!tk::metal::is_available()) {
    throw std::invalid_argument("Metal DLPack import is not available.");
  }
  auto shape = get_shape(nd_array);
  if (nd_array.itemsize() != tk::size_of(src_dtype)) {
    throw std::invalid_argument(
        "Cannot convert Metal DLPack dtype to tiki dtype.");
  }
  auto [storage_size, strides, flags] = get_strided_layout(nd_array, shape);
  auto data_handle = nd_array.data_handle();
  tk::array out(shape, src_dtype, nullptr, {});
  out.set_data(
      tk::allocator::Buffer(data_handle),
      storage_size,
      std::move(strides),
      flags,
      nd_array.byte_offset(),
      [owner = std::move(nd_array)](tk::allocator::Buffer) {});
  out.set_status(tk::array::Status::available);

  if (copy) {
    auto result = tk::astype(out, dst_dtype, true, tk::Device::gpu);
    result.eval();
    return result;
  }
  return out;
}

tk::array nd_array_to_tiki(
    nb::ndarray<nb::ro> nd_array,
    std::optional<tk::Dtype> requested_dtype,
    std::optional<nb::dlpack::dtype> src_dlpack_dtype_override,
    std::optional<bool> copy) {
  auto src_dlpack_dtype = src_dlpack_dtype_override.value_or(nd_array.dtype());
  auto src_tiki_dtype = tiki_dtype_from_dlpack(
      src_dlpack_dtype, "[convert] Cannot convert array to tiki.");
  auto dst_dtype = requested_dtype.value_or(src_tiki_dtype);
  auto device_type = nd_array.device_type();

  // A dtype change requires converting the elements, which cannot be done
  // without a copy.
  bool no_copy = copy.has_value() && !copy.value();
  if (no_copy && dst_dtype != src_tiki_dtype) {
    throw std::invalid_argument(
        "[convert] Cannot convert array to the requested dtype without a "
        "copy.");
  }

  switch (device_type) {
    case nb::device::cpu::value: {
      auto shape = get_shape(nd_array);
      // For copy=None (try to share) and copy=False (must share), attempt a
      // zero-copy adoption of the host buffer first. A copy is passed by value
      // so the source is preserved for the fallback below.
      if (!copy.value_or(false)) {
        if (auto out =
                cpu_nd_array_to_tiki_no_copy(nd_array, shape, dst_dtype)) {
          return *out;
        }
        if (no_copy) {
          throw std::invalid_argument(
              "[convert] Cannot import a CPU array without a copy.");
        }
      }
      // copy=True, or copy=None where adoption was not possible: copy.
      return dispatch_dlpack_dtype(
          src_dlpack_dtype,
          [&]<typename T>(tk::Dtype) {
            return cpu_nd_array_to_tiki<T>(nd_array, shape, dst_dtype);
          },
          "[convert] Cannot convert array to tiki.");
    }
    case nb::device::metal::value: {
      // A Metal buffer can be adopted without a copy only if the active
      // allocator recognizes it.
      bool can_reuse_buffer =
          tk::allocator::can_reuse_alien_buffer(nd_array.data_handle());
      if (no_copy && !can_reuse_buffer) {
        throw std::invalid_argument(
            "[convert] Cannot import a private Metal buffer without a copy.");
      }
      bool should_copy = copy.value_or(false) || dst_dtype != src_tiki_dtype ||
          !can_reuse_buffer;
      return metal_nd_array_to_tiki(
          nd_array, src_tiki_dtype, dst_dtype, should_copy);
    }
    case nb::device::cuda::value:
    case nb::device::cuda_managed::value:
      throw std::invalid_argument("[convert] CUDA import is not supported.");
    default:
      throw std::invalid_argument("[convert] Unsupported device.");
  }
}

template <typename... NDParams>
nb::ndarray<NDParams...> tiki_to_nd_array(
    const tk::array& a,
    std::optional<std::tuple<int, int>> dl_device) {
  auto default_device = tk::metal::is_available()
      ? std::tuple{nb::device::metal::value, 0}
      : std::tuple{nb::device::cpu::value, 0};
  auto [device_type, device_id] = dl_device.value_or(default_device);

  if (device_type == nb::device::cuda::value ||
      device_type == nb::device::cuda_managed::value) {
    throw nb::buffer_error("CUDA DLPack export is not supported.");
  }
  if (device_type != nb::device::cpu::value &&
      device_type != nb::device::metal::value) {
    throw nb::buffer_error(
        "Cannot export tiki array to requested DLPack device.");
  }
  if (device_type == nb::device::metal::value && !tk::metal::is_available()) {
    throw nb::buffer_error("Metal DLPack export is not available.");
  }

  auto arr = a;
  void* data = nullptr;
  uint64_t byte_offset = 0;
  {
    nb::gil_scoped_release nogil;
    arr.eval();
  }
  data = device_type == nb::device::cpu::value ? arr.buffer().raw_ptr()
                                               : arr.buffer().ptr();
  byte_offset = arr.offset();

  std::vector<size_t> shape(arr.shape().begin(), arr.shape().end());
  auto owner = nb::cast(arr);
  return nb::ndarray<NDParams...>(
      data,
      arr.ndim(),
      shape.data(),
      /* owner= */ owner,
      arr.strides().data(),
      tiki_dtype_to_dl_dtype(arr.dtype()),
      device_type,
      device_id,
      '\0',
      byte_offset);
}

nb::ndarray<nb::numpy> tiki_to_np_array(const tk::array& a) {
  if (a.dtype() == tk::bfloat16) {
    throw nb::type_error("bfloat16 arrays cannot be converted to NumPy.");
  }
  return tiki_to_nd_array<nb::numpy>(a, std::tuple{nb::device::cpu::value, 0});
}

nb::ndarray<> tiki_to_dlpack(
    const tk::array& a,
    bool force_copy,
    std::optional<std::tuple<int, int>> dl_device) {
  if (force_copy) {
    return tiki_to_nd_array<>(tk::astype(a, a.dtype(), true), dl_device);
  }
  return tiki_to_nd_array<>(a, dl_device);
}

nb::object to_scalar(tk::array& a) {
  if (a.size() != 1) {
    throw std::invalid_argument(
        "[convert] Only length-1 arrays can be converted to Python scalars.");
  }
  {
    nb::gil_scoped_release nogil;
    a.eval();
  }
  switch (a.dtype()) {
    case tk::bool_:
      return nb::cast(a.item<bool>());
    case tk::uint8:
      return nb::cast(a.item<uint8_t>());
    case tk::uint16:
      return nb::cast(a.item<uint16_t>());
    case tk::uint32:
      return nb::cast(a.item<uint32_t>());
    case tk::uint64:
      return nb::cast(a.item<uint64_t>());
    case tk::int8:
      return nb::cast(a.item<int8_t>());
    case tk::int16:
      return nb::cast(a.item<int16_t>());
    case tk::int32:
      return nb::cast(a.item<int32_t>());
    case tk::int64:
      return nb::cast(a.item<int64_t>());
    case tk::float16:
      return nb::cast(static_cast<float>(a.item<tk::float16_t>()));
    case tk::float32:
      return nb::cast(a.item<float>());
    case tk::bfloat16:
      return nb::cast(static_cast<float>(a.item<tk::bfloat16_t>()));
    case tk::complex64:
      return nb::cast(a.item<std::complex<float>>());
    case tk::float64:
      return nb::cast(a.item<double>());
    default:
      throw nb::type_error("type cannot be converted to Python scalar.");
  }
}

template <typename T, typename U = T>
nb::list to_list(tk::array& a, size_t index, int dim) {
  nb::list pl;
  auto stride = a.strides()[dim];
  for (int i = 0; i < a.shape(dim); ++i) {
    if (dim == a.ndim() - 1) {
      pl.append(static_cast<U>(a.data<T>()[index]));
    } else {
      pl.append(to_list<T, U>(a, index, dim + 1));
    }
    index += stride;
  }
  return pl;
}

nb::object tolist(tk::array& a) {
  if (a.ndim() == 0) {
    return to_scalar(a);
  }
  {
    nb::gil_scoped_release nogil;
    a.eval();
  }
  switch (a.dtype()) {
    case tk::bool_:
      return to_list<bool>(a, 0, 0);
    case tk::uint8:
      return to_list<uint8_t>(a, 0, 0);
    case tk::uint16:
      return to_list<uint16_t>(a, 0, 0);
    case tk::uint32:
      return to_list<uint32_t>(a, 0, 0);
    case tk::uint64:
      return to_list<uint64_t>(a, 0, 0);
    case tk::int8:
      return to_list<int8_t>(a, 0, 0);
    case tk::int16:
      return to_list<int16_t>(a, 0, 0);
    case tk::int32:
      return to_list<int32_t>(a, 0, 0);
    case tk::int64:
      return to_list<int64_t>(a, 0, 0);
    case tk::float16:
      return to_list<tk::float16_t, float>(a, 0, 0);
    case tk::float32:
      return to_list<float>(a, 0, 0);
    case tk::bfloat16:
      return to_list<tk::bfloat16_t, float>(a, 0, 0);
    case tk::float64:
      return to_list<double>(a, 0, 0);
    case tk::complex64:
      return to_list<std::complex<float>>(a, 0, 0);
    default:
      throw nb::type_error("data type cannot be converted to Python list.");
  }
}

template <typename T, typename U>
void fill_vector(T list, std::vector<U>& vals) {
  for (auto l : list) {
    if (nb::isinstance<nb::list>(l)) {
      fill_vector(nb::cast<nb::list>(l), vals);
    } else if (nb::isinstance<nb::tuple>(*list.begin())) {
      fill_vector(nb::cast<nb::tuple>(l), vals);
    } else {
      vals.push_back(nb::cast<U>(l));
    }
  }
}

template <typename T>
PyScalarT validate_shape(
    T list,
    const tk::Shape& shape,
    int idx,
    bool& all_python_primitive_elements,
    bool& has_wide_int) {
  if (idx >= shape.size()) {
    throw std::invalid_argument("Initialization encountered extra dimension.");
  }
  auto s = shape[idx];
  if (nb::len(list) != s) {
    throw std::invalid_argument(
        "Initialization encountered non-uniform length.");
  }

  if (s == 0) {
    return pyfloat;
  }

  PyScalarT type = pybool;
  for (auto l : list) {
    PyScalarT t;
    if (nb::isinstance<nb::list>(l)) {
      t = validate_shape(
          nb::cast<nb::list>(l),
          shape,
          idx + 1,
          all_python_primitive_elements,
          has_wide_int);
    } else if (nb::isinstance<nb::tuple>(*list.begin())) {
      t = validate_shape(
          nb::cast<nb::tuple>(l),
          shape,
          idx + 1,
          all_python_primitive_elements,
          has_wide_int);
    } else if (nb::isinstance<tk::array>(l)) {
      all_python_primitive_elements = false;
      auto arr = nb::cast<tk::array>(l);
      if (arr.ndim() + idx + 1 == shape.size() &&
          std::equal(
              arr.shape().cbegin(),
              arr.shape().cend(),
              shape.cbegin() + idx + 1)) {
        t = pybool;
      } else {
        throw std::invalid_argument(
            "Initialization encountered non-uniform length.");
      }
    } else {
      if (nb::isinstance<nb::bool_>(l)) {
        t = pybool;
      } else if (nb::isinstance<nb::int_>(l)) {
        t = pyint;
        // Match the scalar path, which widens to int64 rather than failing
        // when a python int does not fit in int32.
        auto val = nb::cast<int64_t>(l);
        if (val > std::numeric_limits<int>::max() ||
            val < std::numeric_limits<int>::min()) {
          has_wide_int = true;
        }
      } else if (nb::isinstance<nb::float_>(l)) {
        t = pyfloat;
      } else if (PyComplex_Check(l.ptr())) {
        t = pycomplex;
      } else {
        std::ostringstream msg;
        msg << "Invalid type " << nb::type_name(l.type()).c_str()
            << " received in array initialization.";
        throw std::invalid_argument(msg.str());
      }

      if (idx + 1 != shape.size()) {
        throw std::invalid_argument(
            "Initialization encountered non-uniform length.");
      }
    }
    type = std::max(type, t);
  }
  return type;
}

template <typename T>
void get_shape(T list, tk::Shape& shape) {
  shape.push_back(check_shape_dim(nb::len(list)));
  if (shape.back() > 0) {
    auto l = list.begin();
    if (nb::isinstance<nb::list>(*l)) {
      return get_shape(nb::cast<nb::list>(*l), shape);
    } else if (nb::isinstance<nb::tuple>(*l)) {
      return get_shape(nb::cast<nb::tuple>(*l), shape);
    } else if (nb::isinstance<tk::array>(*l)) {
      auto arr = nb::cast<tk::array>(*l);
      for (int i = 0; i < arr.ndim(); i++) {
        shape.push_back(arr.shape(i));
      }
      return;
    }
  }
}

template <typename T>
tk::array array_from_list_impl(
    T pl,
    const PyScalarT& inferred_type,
    std::optional<tk::Dtype> specified_type,
    const tk::Shape& shape,
    bool has_wide_int) {
  // Make the array
  switch (inferred_type) {
    case pybool: {
      std::vector<bool> vals;
      fill_vector(pl, vals);
      return tk::array(vals.begin(), shape, specified_type.value_or(tk::bool_));
    }
    case pyint: {
      auto dtype =
          specified_type.value_or(has_wide_int ? tk::int64 : tk::int32);
      if (dtype == tk::int64) {
        std::vector<int64_t> vals;
        fill_vector(pl, vals);
        return tk::array(vals.begin(), shape, dtype);
      } else if (dtype == tk::uint64) {
        std::vector<uint64_t> vals;
        fill_vector(pl, vals);
        return tk::array(vals.begin(), shape, dtype);
      } else if (dtype == tk::uint32) {
        std::vector<uint32_t> vals;
        fill_vector(pl, vals);
        return tk::array(vals.begin(), shape, dtype);
      } else if (tk::issubdtype(dtype, tk::inexact)) {
        std::vector<float> vals;
        fill_vector(pl, vals);
        return tk::array(vals.begin(), shape, dtype);
      } else {
        std::vector<int> vals;
        fill_vector(pl, vals);
        return tk::array(vals.begin(), shape, dtype);
      }
    }
    case pyfloat: {
      auto out_type = specified_type.value_or(tk::float32);
      if (out_type == tk::float64) {
        std::vector<double> vals;
        fill_vector(pl, vals);
        return tk::array(vals.begin(), shape, out_type);
      } else {
        std::vector<float> vals;
        fill_vector(pl, vals);
        return tk::array(vals.begin(), shape, out_type);
      }
    }
    case pycomplex: {
      std::vector<std::complex<float>> vals;
      fill_vector(pl, vals);
      return tk::array(
          reinterpret_cast<tk::complex64_t*>(vals.data()),
          shape,
          specified_type.value_or(tk::complex64));
    }
    default: {
      std::ostringstream msg;
      msg << "Should not happen, inferred: " << inferred_type
          << " on subarray made of only python primitive types.";
      throw std::runtime_error(msg.str());
    }
  }
}

template <typename T>
tk::array array_from_list_impl(T pl, std::optional<tk::Dtype> dtype) {
  // Compute the shape
  tk::Shape shape;
  get_shape(pl, shape);

  // Validate the shape and type
  bool all_python_primitive_elements = true;
  bool has_wide_int = false;
  auto type =
      validate_shape(pl, shape, 0, all_python_primitive_elements, has_wide_int);

  if (all_python_primitive_elements) {
    // `pl` does not contain tiki arrays
    return array_from_list_impl(pl, type, dtype, shape, has_wide_int);
  }

  // `pl` contains tiki arrays
  std::vector<tk::array> arrays;
  for (auto l : pl) {
    arrays.push_back(create_array(nb::cast<nb::object>(l), dtype));
  }
  return tk::stack(arrays);
}

tk::array array_from_list(nb::list pl, std::optional<tk::Dtype> dtype) {
  return array_from_list_impl(pl, dtype);
}

tk::array array_from_list(nb::tuple pl, std::optional<tk::Dtype> dtype) {
  return array_from_list_impl(pl, dtype);
}

tk::array create_array(
    nb::object v,
    std::optional<tk::Dtype> t,
    std::optional<bool> copy) {
  if (!nb::isinstance<tk::array>(v) && nb::ndarray_check(v)) {
    using ContigArray = nb::ndarray<nb::ro>;
    ContigArray nd;
    std::optional<nb::dlpack::dtype> nb_dtype;
    // Nanobind does not recognize bfloat16 numpy array:
    // https://github.com/wjakob/nanobind/discussions/560
    if (nb::hasattr(v, "dtype") && v.attr("dtype").equal(nb::str("bfloat16"))) {
      nd = nb::cast<ContigArray>(v.attr("view")("uint16"));
      nb_dtype = nb::dtype<tk::bfloat16_t>();
    } else {
      nd = nb::cast<ContigArray>(v);
    }
    return nd_array_to_tiki(nd, t, nb_dtype, copy);
  }

  if (copy.has_value() && copy.value() == false) {
    throw std::invalid_argument(
        "Unable to avoid copy while creating an array as requested.");
  }

  if (nb::isinstance<nb::bool_>(v)) {
    return tk::array(nb::cast<bool>(v), t.value_or(tk::bool_));
  } else if (nb::isinstance<nb::int_>(v)) {
    auto val = nb::cast<int64_t>(v);
    auto default_type = (val > std::numeric_limits<int>::max() ||
                         val < std::numeric_limits<int>::min())
        ? tk::int64
        : tk::int32;
    return tk::array(val, t.value_or(default_type));
  } else if (nb::isinstance<nb::float_>(v)) {
    auto out_type = t.value_or(tk::float32);
    if (out_type == tk::float64) {
      return tk::array(nb::cast<double>(v), out_type);
    } else {
      return tk::array(nb::cast<float>(v), out_type);
    }
  } else if (PyComplex_Check(v.ptr())) {
    return tk::array(
        static_cast<tk::complex64_t>(nb::cast<std::complex<float>>(v)),
        t.value_or(tk::complex64));
  } else if (nb::isinstance<nb::list>(v)) {
    return array_from_list(nb::cast<nb::list>(v), t);
  } else if (nb::isinstance<nb::tuple>(v)) {
    return array_from_list(nb::cast<nb::tuple>(v), t);
  } else if (nb::isinstance<tk::array>(v)) {
    auto arr = nb::cast<tk::array>(v);
    auto dtype = t.value_or(arr.dtype());
    return tk::astype(arr, dtype, copy.value_or(false));
  } else {
    auto arr = to_array_with_accessor(v);
    return tk::astype(arr, t.value_or(arr.dtype()), copy.value_or(false));
  }
}
