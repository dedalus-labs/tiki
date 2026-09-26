// Copyright © 2023-2024 Apple Inc.
#include <cstdint>
#include <cstring>
#include <sstream>
#include <tuple>

#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/typing.h>

#include "python/src/buffer.h"
#include "python/src/convert.h"
#include "python/src/indexing.h"
#include "python/src/small_vector.h"
#include "python/src/utils.h"
#include "tiki/backend/metal/metal.h"
#include "tiki/utils.h"

#include "tiki/tiki.h"

namespace tk = tiki::core;
namespace nb = nanobind;
using namespace nb::literals;

class ArrayAt {
 public:
  ArrayAt(tk::array x) : x_(std::move(x)) {}
  ArrayAt& set_indices(nb::object indices) {
    initialized_ = true;
    indices_ = indices;
    return *this;
  }
  void check_initialized() {
    if (!initialized_) {
      throw std::invalid_argument(
          "Must give indices to array.at (e.g. `x.at[0].add(4)`).");
    }
  }

  tk::array add(const ScalarOrArray& v) {
    check_initialized();
    return tiki_add_item(x_, indices_, v);
  }
  tk::array subtract(const ScalarOrArray& v) {
    check_initialized();
    return tiki_subtract_item(x_, indices_, v);
  }
  tk::array multiply(const ScalarOrArray& v) {
    check_initialized();
    return tiki_multiply_item(x_, indices_, v);
  }
  tk::array divide(const ScalarOrArray& v) {
    check_initialized();
    return tiki_divide_item(x_, indices_, v);
  }
  tk::array maximum(const ScalarOrArray& v) {
    check_initialized();
    return tiki_maximum_item(x_, indices_, v);
  }
  tk::array minimum(const ScalarOrArray& v) {
    check_initialized();
    return tiki_minimum_item(x_, indices_, v);
  }

 private:
  tk::array x_;
  bool initialized_{false};
  nb::object indices_;
};

class ArrayPythonIterator {
 public:
  ArrayPythonIterator(tk::array x) : idx_(0), x_(std::move(x)) {
    if (x_.ndim() == 0) {
      throw nb::type_error("iter() 0-dimensional array.");
    }
    if (x_.shape(0) > 0 && x_.shape(0) < 10) {
      splits_ = tk::split(x_, x_.shape(0));
    }
  }

  tk::array next() {
    if (idx_ >= x_.shape(0)) {
      throw nb::stop_iteration();
    }

    if (idx_ >= 0 && idx_ < splits_.size()) {
      return tk::squeeze(splits_[idx_++], 0);
    }

    return *(x_.begin() + idx_++);
  }

 private:
  int idx_;
  tk::array x_;
  std::vector<tk::array> splits_;
};

void init_array(nb::module_& m) {
  // Types
  nb::class_<tk::Dtype>(
      m,
      "Dtype",
      R"pbdoc(
      An object to hold the type of a :class:`array`.

      See the :ref:`list of types <data_types>` for more details
      on available data types.
      )pbdoc")
      .def_prop_ro(
          "size", &tk::Dtype::size, R"pbdoc(Size of the type in bytes.)pbdoc")
      .def(
          "__repr__",
          [](const tk::Dtype& t) {
            std::ostringstream os;
            os << "tiki.";
            os << t;
            return os.str();
          })
      .def(
          "__eq__",
          [](const tk::Dtype& t, const nb::object& other) {
            return nb::isinstance<tk::Dtype>(other) &&
                t == nb::cast<tk::Dtype>(other);
          })
      .def("__hash__", [](const tk::Dtype& t) {
        return static_cast<int64_t>(t.val());
      });

  m.attr("bool_") = nb::cast(tk::bool_);
  m.attr("uint8") = nb::cast(tk::uint8);
  m.attr("uint16") = nb::cast(tk::uint16);
  m.attr("uint32") = nb::cast(tk::uint32);
  m.attr("uint64") = nb::cast(tk::uint64);
  m.attr("int8") = nb::cast(tk::int8);
  m.attr("int16") = nb::cast(tk::int16);
  m.attr("int32") = nb::cast(tk::int32);
  m.attr("int64") = nb::cast(tk::int64);
  m.attr("float16") = nb::cast(tk::float16);
  m.attr("float32") = nb::cast(tk::float32);
  m.attr("float64") = nb::cast(tk::float64);
  m.attr("bfloat16") = nb::cast(tk::bfloat16);
  m.attr("complex64") = nb::cast(tk::complex64);
  nb::enum_<tk::Dtype::Category>(
      m,
      "DtypeCategory",
      R"pbdoc(
      Type to hold categories of :class:`dtypes <Dtype>`.

      * :attr:`~tiki.generic`

        * :ref:`bool_ <data_types>`
        * :attr:`~tiki.number`

          * :attr:`~tiki.integer`

            * :attr:`~tiki.unsignedinteger`

              * :ref:`uint8 <data_types>`
              * :ref:`uint16 <data_types>`
              * :ref:`uint32 <data_types>`
              * :ref:`uint64 <data_types>`

            * :attr:`~tiki.signedinteger`

              * :ref:`int8 <data_types>`
              * :ref:`int32 <data_types>`
              * :ref:`int64 <data_types>`

          * :attr:`~tiki.inexact`

            * :attr:`~tiki.floating`

              * :ref:`float16 <data_types>`
              * :ref:`bfloat16 <data_types>`
              * :ref:`float32 <data_types>`
              * :ref:`float64 <data_types>`

            * :attr:`~tiki.complexfloating`

              * :ref:`complex64 <data_types>`

      See also :func:`~tiki.issubdtype`.
      )pbdoc")
      .value("complexfloating", tk::complexfloating)
      .value("floating", tk::floating)
      .value("inexact", tk::inexact)
      .value("signedinteger", tk::signedinteger)
      .value("unsignedinteger", tk::unsignedinteger)
      .value("integer", tk::integer)
      .value("number", tk::number)
      .value("generic", tk::generic)
      .export_values();

  nb::class_<tk::finfo>(
      m,
      "finfo",
      R"pbdoc(
      Get information on floating-point types.
      )pbdoc")
      .def(nb::init<tk::Dtype>())
      .def_ro(
          "bits",
          &tk::finfo::bits,
          R"pbdoc(The number of bits occupied by the type.)pbdoc")
      .def_ro(
          "min",
          &tk::finfo::min,
          R"pbdoc(The smallest representable number.)pbdoc")
      .def_ro(
          "max",
          &tk::finfo::max,
          R"pbdoc(The largest representable number.)pbdoc")
      .def_ro(
          "eps",
          &tk::finfo::eps,
          R"pbdoc(
            The difference between 1.0 and the next smallest
            representable number larger than 1.0.
          )pbdoc")
      .def_ro(
          "smallest_normal",
          &tk::finfo::smallest_normal,
          R"pbdoc(The smallest positive normal number.)pbdoc")
      .def_ro("dtype", &tk::finfo::dtype, R"pbdoc(The :obj:`Dtype`.)pbdoc")
      .def("__repr__", [](const tk::finfo& f) {
        std::ostringstream os;
        os << "finfo("
           << "min=" << f.min << ", max=" << f.max << ", dtype=" << f.dtype
           << ")";
        return os.str();
      });

  nb::class_<tk::iinfo>(
      m,
      "iinfo",
      R"pbdoc(
      Get information on integer types.
      )pbdoc")
      .def(nb::init<tk::Dtype>())
      .def_ro(
          "min",
          &tk::iinfo::min,
          R"pbdoc(The smallest representable number.)pbdoc")
      .def_ro(
          "max",
          &tk::iinfo::max,
          R"pbdoc(The largest representable number.)pbdoc")
      .def_ro("dtype", &tk::iinfo::dtype, R"pbdoc(The :obj:`Dtype`.)pbdoc")
      .def("__repr__", [](const tk::iinfo& i) {
        std::ostringstream os;
        os << "iinfo("
           << "min=" << i.min << ", max=" << i.max << ", dtype=" << i.dtype
           << ")";
        return os.str();
      });

  nb::class_<ArrayAt>(
      m,
      "ArrayAt",
      R"pbdoc(
      A helper object to apply updates at specific indices.
      )pbdoc",
      nb::pooled(/* capacity = */ 128))
      .def("__getitem__", &ArrayAt::set_indices, "indices"_a.none())
      .def("add", &ArrayAt::add, "value"_a)
      .def("subtract", &ArrayAt::subtract, "value"_a)
      .def("multiply", &ArrayAt::multiply, "value"_a)
      .def("divide", &ArrayAt::divide, "value"_a)
      .def("maximum", &ArrayAt::maximum, "value"_a)
      .def("minimum", &ArrayAt::minimum, "value"_a);

  nb::class_<ArrayLike>(
      m,
      "ArrayLike",
      R"pbdoc(
        Any Python object which has an ``__tiki__array__`` method that
        returns an :obj:`array`.
      )pbdoc")
      .def(nb::init_implicit<nb::object>());

  nb::class_<ArrayPythonIterator>(
      m,
      "ArrayIterator",
      R"pbdoc(
      A helper object to iterate over the 1st dimension of an array.
      )pbdoc")
      .def("__next__", &ArrayPythonIterator::next)
      .def("__iter__", [](const ArrayPythonIterator& it) { return it; });

  // Install buffer protocol functions
  PyType_Slot array_slots[] = {
      {Py_bf_getbuffer, (void*)getbuffer},
      {Py_bf_releasebuffer, (void*)releasebuffer},
      {0, nullptr}};

  nb::class_<tk::array>(
      m,
      "array",
      R"pbdoc(An N-dimensional array object.)pbdoc",
      nb::type_slots(array_slots),
      nb::is_weak_referenceable(),
      nb::pooled(/* capacity = */ 128))
      .def(
          "__init__",
          [](tk::array* aptr, nb::object v, std::optional<tk::Dtype> t) {
            new (aptr) tk::array(create_array(v, t, true));
          },
          "val"_a,
          "dtype"_a = nb::none(),
          nb::sig(
              "def __init__(self: array, val: scalar | list | tuple | DLPackCompatible | array, dtype: Dtype | None = None)"))
      .def_prop_ro(
          "size",
          &tk::array::size,
          R"pbdoc(Number of elements in the array.)pbdoc")
      .def_prop_ro(
          "ndim", &tk::array::ndim, R"pbdoc(The array's dimension.)pbdoc")
      .def_prop_ro(
          "itemsize",
          &tk::array::itemsize,
          R"pbdoc(The size of the array's datatype in bytes.)pbdoc")
      .def_prop_ro(
          "nbytes",
          &tk::array::nbytes,
          R"pbdoc(The number of bytes in the array.)pbdoc")
      .def_prop_ro(
          "shape",
          [](const tk::array& a) { return nb::cast(a.shape()); },
          nb::sig("def shape(self) -> tuple[int, ...]"),
          R"pbdoc(
          The shape of the array as a Python tuple.

          Returns:
            tuple(int): A tuple containing the sizes of each dimension.
        )pbdoc")
      .def_prop_ro(
          "dtype",
          &tk::array::dtype,
          R"pbdoc(
            The array's :class:`Dtype`.
          )pbdoc")
      .def_prop_ro(
          "strides",
          [](tk::array& a) {
            a.eval();
            return nb::cast(a.strides());
          },
          nb::sig("def strides(self) -> tuple[int, ...]"),
          R"pbdoc(
          The element strides of the evaluated array, one per dimension.

          A lazy view is evaluated first, because its strides are only known
          once it has storage.
        )pbdoc")
      .def_prop_ro(
          "offset",
          [](tk::array& a) {
            a.eval();
            return a.offset() / static_cast<int64_t>(a.itemsize());
          },
          R"pbdoc(
          The element offset of the evaluated array into its storage, in the
          units :func:`as_strided` takes.
        )pbdoc")
      .def_prop_ro(
          "real",
          [](const tk::array& a) { return tk::real(a); },
          R"pbdoc(
            The real part of a complex array.
          )pbdoc")
      .def_prop_ro(
          "imag",
          [](const tk::array& a) { return tk::imag(a); },
          R"pbdoc(
            The imaginary part of a complex array.
          )pbdoc")
      .def(
          "item",
          &to_scalar,
          nb::sig("def item(self) -> scalar"),
          R"pbdoc(
            Access the value of a scalar array.

            Returns:
                Standard Python scalar.
          )pbdoc")
      .def(
          "tolist",
          &tolist,
          nb::sig("def tolist(self) -> list_or_scalar"),
          R"pbdoc(
            Convert the array to a Python :class:`list`.

            Returns:
                list: The Python list.

                If the array is a scalar then a standard Python scalar is returned.

                If the array has more than one dimension then the result is a nested
                list of lists.

                The value type of the list corresponding to the last dimension is either
                ``bool``, ``int`` or ``float`` depending on the ``dtype`` of the array.
          )pbdoc")
      .def(
          "astype",
          [](const tk::array& a, tk::Dtype dtype, tk::StreamOrDevice s) {
            return tk::astype(a, dtype, s);
          },
          "dtype"_a,
          "stream"_a = nb::none(),
          R"pbdoc(
            Cast the array to a specified type.

            Args:
                dtype (Dtype): Type to which the array is cast.
                stream (Stream): Stream (or device) for the operation.

            Returns:
                array: The array with type ``dtype``.
          )pbdoc")
      .def(
          "__array_namespace__",
          [](const tk::array& a,
             const std::optional<std::string>& api_version) {
            if (api_version) {
              throw std::invalid_argument(
                  "Explicitly specifying api_version is not yet implemented.");
            }
            return nb::module_::import_("tiki");
          },
          "api_version"_a = nb::none(),
          R"pbdoc(
            Returns an object that has all the array API functions on it.

            See the `Python array API <https://data-apis.org/array-api/latest/index.html>`_
            for more information.

            Args:
                api_version (str, optional): String representing the version
                  of the array API spec to return. Default: ``None``.

            Returns:
                out (Any): An object representing the array API namespace.
          )pbdoc")
      .def("__getitem__", tiki_get_item, nb::arg().none())
      .def("__setitem__", tiki_set_item, nb::arg().none(), nb::arg())
      .def_prop_ro(
          "at",
          [](const tk::array& a) { return ArrayAt(a); },
          R"pbdoc(
            Used to apply updates at the given indices.

            .. note::

               Regular in-place updates map to assignment. For instance ``x[idx] += y``
               maps to ``x[idx] = x[idx] + y``. As a result, assigning to the
               same index ignores all but one update. Using ``x.at[idx].add(y)``
               will correctly apply all updates to all indices.

            .. list-table::
               :header-rows: 1

               * - array.at syntax
                 - In-place syntax
               * - ``x = x.at[idx].add(y)``
                 - ``x[idx] += y``
               * - ``x = x.at[idx].subtract(y)``
                 - ``x[idx] -= y``
               * - ``x = x.at[idx].multiply(y)``
                 - ``x[idx] *= y``
               * - ``x = x.at[idx].divide(y)``
                 - ``x[idx] /= y``
               * - ``x = x.at[idx].maximum(y)``
                 - ``x[idx] = tk.maximum(x[idx], y)``
               * - ``x = x.at[idx].minimum(y)``
                 - ``x[idx] = tk.minimum(x[idx], y)``

            Example:
                >>> a = tk.array([0, 0])
                >>> idx = tk.array([0, 1, 0, 1])
                >>> a[idx] += 1
                >>> a
                array([1, 1], dtype=int32)
                >>>
                >>> a = tk.array([0, 0])
                >>> a.at[idx].add(1)
                array([2, 2], dtype=int32)
          )pbdoc")
      .def(
          "__len__",
          [](const tk::array& a) {
            if (a.ndim() == 0) {
              throw nb::type_error("len() 0-dimensional array.");
            }
            return a.shape(0);
          })
      .def(
          "__iter__", [](const tk::array& a) { return ArrayPythonIterator(a); })
      .def(
          "__getstate__",
          [](const tk::array& a) {
            auto nd = (a.dtype() == tk::bfloat16)
                ? tiki_to_np_array(tk::view(a, tk::uint16))
                : tiki_to_np_array(a);
            return nb::make_tuple(nd, static_cast<uint8_t>(a.dtype().val()));
          })
      .def(
          "__setstate__",
          [](tk::array& arr, const nb::tuple& state) {
            if (nb::len(state) != 2) {
              throw std::invalid_argument(
                  "Invalid pickle state: expected (ndarray, Dtype::Val)");
            }
            using ND = nb::ndarray<nb::ro>;
            ND nd = nb::cast<ND>(state[0]);
            auto val = static_cast<tk::Dtype::Val>(nb::cast<uint8_t>(state[1]));
            if (val == tk::Dtype::Val::bfloat16) {
              auto owner = nb::handle(state[0].ptr());
              new (&arr) tk::array(nd_array_to_tiki(
                  ND(nd.data(),
                     nd.ndim(),
                     reinterpret_cast<const size_t*>(nd.shape_ptr()),
                     owner,
                     nullptr,
                     nb::dtype<tk::bfloat16_t>()),
                  tk::bfloat16));
            } else {
              new (&arr) tk::array(nd_array_to_tiki(nd, std::nullopt));
            }
          })
      .def(
          "__dlpack__",
          [](const tk::array& a,
             nb::object,
             nb::object,
             std::optional<std::tuple<int, int>> dl_device,
             std::optional<bool> copy) {
            return tiki_to_dlpack(a, copy.value_or(false), dl_device);
          },
          nb::kw_only(),
          "stream"_a = nb::none(),
          "max_version"_a = nb::none(),
          "dl_device"_a = nb::none(),
          "copy"_a = nb::none())
      .def(
          "__dlpack_device__",
          [](const tk::array& a) {
            // See
            // https://github.com/dmlc/dlpack/blob/5c210da409e7f1e51ddf445134a4376fdbd70d7d/include/dlpack/dlpack.h#L74
            if (tk::metal::is_available()) {
              return nb::make_tuple(8, 0);
            } else {
              // CPU device
              return nb::make_tuple(1, 0);
            }
          })
      .def(
          "__array__",
          [](const tk::array& self, nb::object dtype, nb::object copy) {
            return tiki_to_np_array(self);
          },
          "dtype"_a = nb::none(),
          "copy"_a = nb::none())
      .def("__copy__", [](const tk::array& self) { return tk::array(self); })
      .def(
          "__deepcopy__",
          [](const tk::array& self, nb::dict) { return tk::array(self); },
          "memo"_a)
      .def(
          "__add__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("addition", v);
            }
            auto b = to_array(v, a.dtype());
            return tk::add(a, b);
          },
          "other"_a)
      .def(
          "__iadd__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace addition", v);
            }
            a.overwrite_descriptor(tk::add(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__radd__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("addition", v);
            }
            return tk::add(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__sub__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("subtraction", v);
            }
            return tk::subtract(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__isub__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace subtraction", v);
            }
            a.overwrite_descriptor(tk::subtract(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rsub__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("subtraction", v);
            }
            return tk::subtract(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__mul__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("multiplication", v);
            }
            return tk::multiply(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__imul__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace multiplication", v);
            }
            a.overwrite_descriptor(tk::multiply(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rmul__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("multiplication", v);
            }
            return tk::multiply(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__truediv__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return tk::divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__itruediv__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace division", v);
            }
            if (!tk::issubdtype(a.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "In place division cannot cast to non-floating point type.");
            }
            a.overwrite_descriptor(divide(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rtruediv__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return tk::divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__div__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return tk::divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rdiv__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return tk::divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__floordiv__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("floor division", v);
            }
            return tk::floor_divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ifloordiv__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace floor division", v);
            }
            a.overwrite_descriptor(tk::floor_divide(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rfloordiv__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("floor division", v);
            }
            auto b = to_array(v, a.dtype());
            return tk::floor_divide(b, a);
          },
          "other"_a)
      .def(
          "__mod__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("modulus", v);
            }
            return tk::remainder(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__imod__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace modulus", v);
            }
            a.overwrite_descriptor(tk::remainder(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rmod__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("modulus", v);
            }
            return tk::remainder(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__eq__",
          [](const tk::array& a,
             const ScalarOrArray& v) -> std::variant<tk::array, bool> {
            if (!is_comparable_with_array(v)) {
              return false;
            }
            return tk::equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__lt__",
          [](const tk::array& a, const ScalarOrArray v) -> tk::array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("less than", v);
            }
            return tk::less(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__le__",
          [](const tk::array& a, const ScalarOrArray v) -> tk::array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("less than or equal", v);
            }
            return tk::less_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__gt__",
          [](const tk::array& a, const ScalarOrArray v) -> tk::array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("greater than", v);
            }
            return tk::greater(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ge__",
          [](const tk::array& a, const ScalarOrArray v) -> tk::array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("greater than or equal", v);
            }
            return tk::greater_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ne__",
          [](const tk::array& a,
             const ScalarOrArray v) -> std::variant<tk::array, bool> {
            if (!is_comparable_with_array(v)) {
              return true;
            }
            return tk::not_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def("__neg__", [](const tk::array& a) { return -a; })
      .def("__bool__", [](tk::array& a) { return nb::bool_(to_scalar(a)); })
      .def(
          "__repr__",
          [](tk::array& a) {
            nb::gil_scoped_release nogil;
            std::ostringstream os;
            os << a;
            return os.str();
          })
      .def(
          "__matmul__",
          [](const tk::array& a, tk::array& other) {
            return tk::matmul(a, other);
          },
          "other"_a)
      .def(
          "__imatmul__",
          [](tk::array& a, tk::array& other) -> tk::array& {
            a.overwrite_descriptor(tk::matmul(a, other));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__pow__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("power", v);
            }
            return tk::power(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rpow__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("power", v);
            }
            return tk::power(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__ipow__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace power", v);
            }
            a.overwrite_descriptor(tk::power(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__invert__",
          [](const tk::array& a) {
            if (tk::issubdtype(a.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise inversion.");
            }
            if (a.dtype() == tk::bool_) {
              return tk::logical_not(a);
            }
            return tk::bitwise_invert(a);
          })
      .def(
          "__and__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("bitwise and", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise and.");
            }
            return tk::bitwise_and(a, b);
          },
          "other"_a)
      .def(
          "__iand__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace bitwise and", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise and.");
            }
            a.overwrite_descriptor(tk::bitwise_and(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__or__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("bitwise or", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise or.");
            }
            return tk::bitwise_or(a, b);
          },
          "other"_a)
      .def(
          "__ior__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace bitwise or", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise or.");
            }
            a.overwrite_descriptor(tk::bitwise_or(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__lshift__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("left shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with left shift.");
            }
            return tk::left_shift(a, b);
          },
          "other"_a)
      .def(
          "__ilshift__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace left shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with left shift.");
            }
            a.overwrite_descriptor(tk::left_shift(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rshift__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("right shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with right shift.");
            }
            return tk::right_shift(a, b);
          },
          "other"_a)
      .def(
          "__irshift__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace right shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with right shift.");
            }
            a.overwrite_descriptor(tk::right_shift(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__xor__",
          [](const tk::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("bitwise xor", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise xor.");
            }
            return tk::bitwise_xor(a, b);
          },
          "other"_a)
      .def(
          "__ixor__",
          [](tk::array& a, const ScalarOrArray v) -> tk::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace bitwise xor", v);
            }
            auto b = to_array(v, a.dtype());
            if (tk::issubdtype(a.dtype(), tk::inexact) ||
                tk::issubdtype(b.dtype(), tk::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed bitwise xor.");
            }
            a.overwrite_descriptor(tk::bitwise_xor(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def("__int__", [](tk::array& a) { return nb::int_(to_scalar(a)); })
      .def("__float__", [](tk::array& a) { return nb::float_(to_scalar(a)); })
      .def(
          "__complex__",
          [](tk::array& a) {
            return nb::cast<std::complex<double>>(to_scalar(a));
          })
      .def(
          "__index__",
          [](tk::array& a) {
            if (!tk::issubdtype(a.dtype(), tk::integer) || a.ndim() != 0) {
              throw nb::type_error(
                  "Only 0-dimensional integer arrays can be converted to an index.");
            }
            return nb::int_(to_scalar(a));
          })
      .def(
          "__bytes__",
          [](tk::array& a) {
            a.eval();
            return nb::bytes(
                reinterpret_cast<const char*>(a.data<void>()), a.nbytes());
          })
      .def(
          "__format__",
          [](tk::array& a, nb::object format_spec) {
            if (nb::len(nb::str(format_spec)) > 0 && a.ndim() > 0) {
              throw nb::type_error(
                  "unsupported format string passed to tk.array.__format__");
            } else if (a.ndim() == 0) {
              auto obj = to_scalar(a);
              return nb::cast<std::string>(
                  nb::handle(PyObject_Format(obj.ptr(), format_spec.ptr())));
            } else {
              nb::gil_scoped_release nogil;
              std::ostringstream os;
              os << a;
              return os.str();
            }
          })
      .def(
          "flatten",
          [](const tk::array& a,
             int start_axis,
             int end_axis,
             const tk::StreamOrDevice& s) {
            return tk::flatten(a, start_axis, end_axis, s);
          },
          "start_axis"_a = 0,
          "end_axis"_a = -1,
          nb::kw_only(),
          "stream"_a = nb::none(),
          R"pbdoc(
            See :func:`flatten`.
          )pbdoc")
      .def(
          "reshape",
          [](const tk::array& a, nb::args shape_, tk::StreamOrDevice s) {
            tk::Shape shape;
            if (!nb::isinstance<int>(shape_[0])) {
              shape = nb::cast<tk::Shape>(shape_[0]);
            } else {
              shape = nb::cast<tk::Shape>(shape_);
            }
            return tk::reshape(a, std::move(shape), s);
          },
          "shape"_a,
          "stream"_a = nb::none(),
          R"pbdoc(
            Equivalent to :func:`reshape` but the shape can be passed either as a
            :obj:`tuple` or as separate arguments.

            See :func:`reshape` for full documentation.
          )pbdoc")
      .def(
          "squeeze",
          [](const tk::array& a,
             const IntOrVec& v,
             const tk::StreamOrDevice& s) {
            if (std::holds_alternative<std::monostate>(v)) {
              return tk::squeeze(a, s);
            } else if (auto pv = std::get_if<int>(&v); pv) {
              return tk::squeeze(a, *pv, s);
            } else {
              return tk::squeeze(a, std::get<std::vector<int>>(v), s);
            }
          },
          "axis"_a = nb::none(),
          nb::kw_only(),
          "stream"_a = nb::none(),
          R"pbdoc(
            See :func:`squeeze`.
          )pbdoc")
      .def(
          "abs",
          &tk::abs,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`abs`.")
      .def(
          "__abs__",
          [](const tk::array& a) { return tk::abs(a); },
          "See :func:`abs`.")
      .def(
          "square",
          &tk::square,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`square`.")
      .def(
          "sqrt",
          &tk::sqrt,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`sqrt`.")
      .def(
          "rsqrt",
          &tk::rsqrt,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`rsqrt`.")
      .def(
          "reciprocal",
          &tk::reciprocal,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`reciprocal`.")
      .def(
          "exp",
          &tk::exp,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`exp`.")
      .def(
          "log",
          &tk::log,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log`.")
      .def(
          "log2",
          &tk::log2,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log2`.")
      .def(
          "log10",
          &tk::log10,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log10`.")
      .def(
          "sin",
          &tk::sin,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`sin`.")
      .def(
          "cos",
          &tk::cos,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`cos`.")
      .def(
          "log1p",
          &tk::log1p,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log1p`.")
      .def(
          "all",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            return tk::all(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`all`.")
      .def(
          "any",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            return tk::any(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`any`.")
      .def(
          "moveaxis",
          &tk::moveaxis,
          "source"_a,
          "destination"_a,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`moveaxis`.")
      .def(
          "swapaxes",
          &tk::swapaxes,
          "axis1"_a,
          "axis2"_a,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`swapaxes`.")
      .def(
          "transpose",
          [](const tk::array& a, nb::args axes_, tk::StreamOrDevice s) {
            if (axes_.size() == 0) {
              return tk::transpose(a, s);
            }
            std::vector<int> axes;
            if (!nb::isinstance<int>(axes_[0])) {
              axes = nb::cast<std::vector<int>>(axes_[0]);
            } else {
              axes = nb::cast<std::vector<int>>(axes_);
            }
            return tk::transpose(a, axes, s);
          },
          "axes"_a,
          "stream"_a = nb::none(),
          R"pbdoc(
            Equivalent to :func:`transpose` but the axes can be passed either as
            a tuple or as separate arguments.

            See :func:`transpose` for full documentation.
          )pbdoc")
      .def_prop_ro(
          "T",
          [](const tk::array& a) { return tk::transpose(a); },
          "Equivalent to calling ``self.transpose()`` with no arguments.")
      .def(
          "sum",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            return tk::sum(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`sum`.")
      .def(
          "prod",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            return tk::prod(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`prod`.")
      .def(
          "min",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            return tk::min(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`min`.")
      .def(
          "max",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            return tk::max(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`max`.")
      .def(
          "logcumsumexp",
          [](const tk::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             tk::StreamOrDevice s) {
            if (axis) {
              return tk::logcumsumexp(a, *axis, reverse, inclusive, s);
            } else {
              return tk::logcumsumexp(a, reverse, inclusive, s);
            }
          },
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
          "See :func:`logcumsumexp`.")
      .def(
          "logsumexp",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            return tk::logsumexp(
                a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`logsumexp`.")
      .def(
          "mean",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            return tk::mean(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`mean`.")
      .def(
          "std",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             int ddof,
             std::optional<int> correction,
             tk::StreamOrDevice s) {
            ddof = correction.value_or(ddof);
            return tk::std(
                a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          "ddof"_a = 0,
          nb::kw_only(),
          "correction"_a = nb::none(),
          "stream"_a = nb::none(),
          "See :func:`std`.")
      .def(
          "var",
          [](const tk::array& a,
             const IntOrVec& axis,
             bool keepdims,
             int ddof,
             std::optional<int> correction,
             tk::StreamOrDevice s) {
            ddof = correction.value_or(ddof);
            return tk::var(
                a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          "ddof"_a = 0,
          nb::kw_only(),
          "correction"_a = nb::none(),
          "stream"_a = nb::none(),
          "See :func:`var`.")
      .def(
          "split",
          [](const tk::array& a,
             const std::variant<int, tk::Shape>& indices_or_sections,
             int axis,
             tk::StreamOrDevice s) {
            if (auto pv = std::get_if<int>(&indices_or_sections); pv) {
              return tk::split(a, *pv, axis, s);
            } else {
              return tk::split(
                  a, std::get<tk::Shape>(indices_or_sections), axis, s);
            }
          },
          "indices_or_sections"_a,
          "axis"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`split`.")
      .def(
          "argmin",
          [](const tk::array& a,
             std::optional<int> axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            if (axis) {
              return tk::argmin(a, *axis, keepdims, s);
            } else {
              return tk::argmin(a, keepdims, s);
            }
          },
          "axis"_a = std::nullopt,
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`argmin`.")
      .def(
          "argmax",
          [](const tk::array& a,
             std::optional<int> axis,
             bool keepdims,
             tk::StreamOrDevice s) {
            if (axis) {
              return tk::argmax(a, *axis, keepdims, s);
            } else {
              return tk::argmax(a, keepdims, s);
            }
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`argmax`.")
      .def(
          "cumsum",
          [](const tk::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             tk::StreamOrDevice s) {
            if (axis) {
              return tk::cumsum(a, *axis, reverse, inclusive, s);
            } else {
              return tk::cumsum(a, reverse, inclusive, s);
            }
          },
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
          "See :func:`cumsum`.")
      .def(
          "cumprod",
          [](const tk::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             tk::StreamOrDevice s) {
            if (axis) {
              return tk::cumprod(a, *axis, reverse, inclusive, s);
            } else {
              return tk::cumprod(a, reverse, inclusive, s);
            }
          },
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
          "See :func:`cumprod`.")
      .def(
          "cummax",
          [](const tk::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             tk::StreamOrDevice s) {
            if (axis) {
              return tk::cummax(a, *axis, reverse, inclusive, s);
            } else {
              return tk::cummax(a, reverse, inclusive, s);
            }
          },
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
          "See :func:`cummax`.")
      .def(
          "cummin",
          [](const tk::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             tk::StreamOrDevice s) {
            if (axis) {
              return tk::cummin(a, *axis, reverse, inclusive, s);
            } else {
              return tk::cummin(a, reverse, inclusive, s);
            }
          },
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
          "See :func:`cummin`.")
      .def(
          "round",
          [](const tk::array& a, int decimals, tk::StreamOrDevice s) {
            return tk::round(a, decimals, s);
          },
          "decimals"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`round`.")
      .def(
          "diagonal",
          [](const tk::array& a,
             int offset,
             int axis1,
             int axis2,
             tk::StreamOrDevice s) {
            return tk::diagonal(a, offset, axis1, axis2, s);
          },
          "offset"_a = 0,
          "axis1"_a = 0,
          "axis2"_a = 1,
          "stream"_a = nb::none(),
          "See :func:`diagonal`.")
      .def(
          "diag",
          [](const tk::array& a, int k, tk::StreamOrDevice s) {
            return tk::diag(a, k, s);
          },
          "k"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          R"pbdoc(
            Extract a diagonal or construct a diagonal matrix.
        )pbdoc")
      .def(
          "conj",
          [](const tk::array& a, tk::StreamOrDevice s) {
            return tk::conjugate(to_array(a), s);
          },
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`conj`.")
      .def(
          "view",
          [](const ScalarOrArray& a,
             const tk::Dtype& dtype,
             tk::StreamOrDevice s) { return tk::view(to_array(a), dtype, s); },
          "dtype"_a,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`view`.");
}
