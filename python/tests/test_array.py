# Copyright © 2023-2024 Apple Inc.

import operator
import os
import pickle
import platform
import sys
import unittest
import warnings
import weakref
from copy import copy, deepcopy
from itertools import permutations

import tiki as tk
import tiki_tests
import numpy as np

try:
    import tensorflow as tf

    has_tf = True
except ImportError:
    has_tf = False


try:
    import torch

    torch_version = [int(v) for v in torch.__version__.split("+")[0].split(".")]
    is_torch_212 = torch_version[0] > 2 or (
        torch_version[0] == 2 and torch_version[1] >= 12
    )
    has_torch_mps = (
        is_torch_212
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
except ImportError:
    torch = None
    has_torch_mps = False


class TestVersion(tiki_tests.TIKITestCase):
    def test_version(self):
        v = tk.__version__
        vnums = v.split(".")
        self.assertGreaterEqual(len(vnums), 3)
        v = ".".join(str(int(vn)) for vn in vnums[:3])
        self.assertEqual(v, tk.__version__[: len(v)])


class TestArrayNamespsceInfo(tiki_tests.TIKITestCase):
    def test(self):
        namespace = tk.__array_namespace_info__()

        self.assertEqual(namespace.default_device(), tk.default_device())
        self.assertEqual(
            namespace.default_dtypes(),
            {
                "real floating": tk.float32,
                "complex floating": tk.complex64,
                "integral": tk.int32,
                "indexing": tk.int32,
            },
        )
        self.assertEqual(
            namespace.dtypes(device=tk.Device(tk.cpu), kind="real floating"),
            {"float32": tk.float32, "float64": tk.float64},
        )
        if tk.is_available(tk.gpu):
            self.assertEqual(
                namespace.dtypes(device=tk.Device(tk.gpu), kind="real floating"),
                {"float32": tk.float32},
            )
        self.assertEqual(
            namespace.dtypes(kind=("bool", "complex floating")),
            {"bool": tk.bool_, "complex64": tk.complex64},
        )
        with self.assertRaises(ValueError):
            namespace.dtypes(kind="invalid")


class TestDtypes(tiki_tests.TIKITestCase):
    def test_dtypes(self):
        self.assertEqual(tk.bool_.size, 1)
        self.assertEqual(tk.uint8.size, 1)
        self.assertEqual(tk.uint16.size, 2)
        self.assertEqual(tk.uint32.size, 4)
        self.assertEqual(tk.uint64.size, 8)
        self.assertEqual(tk.int8.size, 1)
        self.assertEqual(tk.int16.size, 2)
        self.assertEqual(tk.int32.size, 4)
        self.assertEqual(tk.int64.size, 8)
        self.assertEqual(tk.float16.size, 2)
        self.assertEqual(tk.float32.size, 4)
        self.assertEqual(tk.bfloat16.size, 2)
        self.assertEqual(tk.complex64.size, 8)

        self.assertEqual(str(tk.bool_), "tiki.bool")
        self.assertEqual(str(tk.uint8), "tiki.uint8")
        self.assertEqual(str(tk.uint16), "tiki.uint16")
        self.assertEqual(str(tk.uint32), "tiki.uint32")
        self.assertEqual(str(tk.uint64), "tiki.uint64")
        self.assertEqual(str(tk.int8), "tiki.int8")
        self.assertEqual(str(tk.int16), "tiki.int16")
        self.assertEqual(str(tk.int32), "tiki.int32")
        self.assertEqual(str(tk.int64), "tiki.int64")
        self.assertEqual(str(tk.float16), "tiki.float16")
        self.assertEqual(str(tk.float32), "tiki.float32")
        self.assertEqual(str(tk.bfloat16), "tiki.bfloat16")
        self.assertEqual(str(tk.complex64), "tiki.complex64")

    def test_scalar_conversion(self):
        dtypes = [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "complex64",
        ]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                x = np.array(2, dtype=getattr(np, dtype))
                y = np.min(x)

                self.assertEqual(x.dtype, y.dtype)
                self.assertTupleEqual(x.shape, y.shape)

                z = tk.array(y)
                self.assertEqual(np.array(z), x)
                self.assertEqual(np.array(z), y)
                self.assertEqual(z.dtype, getattr(tk, dtype))
                self.assertListEqual(list(z.shape), list(x.shape))
                self.assertListEqual(list(z.shape), list(y.shape))

    def test_index_conversion(self):
        for dtype in [
            tk.uint8,
            tk.uint16,
            tk.uint32,
            tk.uint64,
            tk.int8,
            tk.int16,
            tk.int32,
            tk.int64,
        ]:
            with self.subTest(dtype=dtype):
                self.assertEqual(operator.index(tk.array(2, dtype)), 2)
                self.assertEqual(list(range(tk.array(3, dtype))), [0, 1, 2])

    def test_index_conversion_invalid(self):
        for dtype in [tk.float16, tk.float32, tk.bfloat16, tk.complex64, tk.bool_]:
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    operator.index(tk.array(2, dtype))

                with self.assertRaises(TypeError):
                    list(range(tk.array(3, dtype)))

    def test_finfo(self):
        with self.assertRaises(ValueError):
            tk.finfo(tk.int32)

        self.assertEqual(tk.finfo(tk.float32).min, np.finfo(np.float32).min)
        self.assertEqual(tk.finfo(tk.float32).max, np.finfo(np.float32).max)
        self.assertEqual(tk.finfo(tk.float32).eps, np.finfo(np.float32).eps)
        self.assertEqual(tk.finfo(tk.float32).bits, np.finfo(np.float32).bits)
        self.assertEqual(
            tk.finfo(tk.float32).smallest_normal,
            float(np.finfo(np.float32).smallest_normal),
        )
        self.assertEqual(tk.finfo(tk.float32).dtype, tk.float32)

        self.assertEqual(tk.finfo(tk.float16).min, np.finfo(np.float16).min)
        self.assertEqual(tk.finfo(tk.float16).max, np.finfo(np.float16).max)
        self.assertEqual(tk.finfo(tk.float16).eps, np.finfo(np.float16).eps)
        self.assertEqual(tk.finfo(tk.float16).bits, np.finfo(np.float16).bits)
        self.assertEqual(
            tk.finfo(tk.float16).smallest_normal,
            float(np.finfo(np.float16).smallest_normal),
        )
        self.assertEqual(tk.finfo(tk.float16).dtype, tk.float16)

        # bfloat16 has no numpy equivalent; check against known IEEE values.
        self.assertEqual(tk.finfo(tk.bfloat16).bits, 16)
        self.assertAlmostEqual(
            tk.finfo(tk.bfloat16).smallest_normal, 2.0**-126, places=40
        )

        # finfo of a complex type reports its real component (array API).
        self.assertEqual(tk.finfo(tk.complex64).dtype, tk.float32)
        self.assertEqual(tk.finfo(tk.complex64).bits, 32)

    def test_iinfo(self):
        with self.assertRaises(ValueError):
            tk.iinfo(tk.float32)

        self.assertEqual(tk.iinfo(tk.int32).min, np.iinfo(np.int32).min)
        self.assertEqual(tk.iinfo(tk.int32).max, np.iinfo(np.int32).max)
        self.assertEqual(tk.iinfo(tk.int32).dtype, tk.int32)

        self.assertEqual(tk.iinfo(tk.uint32).min, np.iinfo(np.uint32).min)
        self.assertEqual(tk.iinfo(tk.uint32).max, np.iinfo(np.uint32).max)
        self.assertEqual(tk.iinfo(tk.int8).dtype, tk.int8)

    def test_result_type(self):
        self.assertEqual(tk.result_type(tk.int8, tk.int16), tk.int16)
        self.assertEqual(tk.result_type(tk.float32, tk.float64), tk.float64)
        # Accepts arrays as well as dtypes, and more than two inputs.
        self.assertEqual(
            tk.result_type(tk.array([1], dtype=tk.int8), tk.int16, tk.int32),
            tk.int32,
        )
        self.assertEqual(
            tk.result_type(tk.array(1.0), tk.array(1, dtype=tk.int32)),
            tk.float32,
        )
        with self.assertRaises(ValueError):
            tk.result_type()

    def test_can_cast(self):
        self.assertTrue(tk.can_cast(tk.int8, tk.int16))
        self.assertFalse(tk.can_cast(tk.int16, tk.int8))
        self.assertTrue(tk.can_cast(tk.float32, tk.float64))
        self.assertFalse(tk.can_cast(tk.float64, tk.float32))
        self.assertTrue(tk.can_cast(tk.uint8, tk.int16))
        self.assertFalse(tk.can_cast(tk.uint16, tk.int16))
        # Accepts an array for the source.
        self.assertTrue(tk.can_cast(tk.array([1, 2, 3], dtype=tk.int8), tk.int32))

    def test_isdtype(self):
        self.assertTrue(tk.isdtype(tk.int32, tk.int32))
        self.assertFalse(tk.isdtype(tk.int32, tk.int16))
        self.assertTrue(tk.isdtype(tk.int32, "signed integer"))
        self.assertTrue(tk.isdtype(tk.uint8, "unsigned integer"))
        self.assertFalse(tk.isdtype(tk.uint8, "signed integer"))
        self.assertTrue(tk.isdtype(tk.int16, "integral"))
        self.assertTrue(tk.isdtype(tk.float32, "real floating"))
        self.assertTrue(tk.isdtype(tk.complex64, "complex floating"))
        self.assertTrue(tk.isdtype(tk.bool_, "bool"))
        self.assertFalse(tk.isdtype(tk.bool_, "numeric"))
        self.assertTrue(tk.isdtype(tk.float32, "numeric"))
        # Tuple of kinds (any match).
        self.assertTrue(tk.isdtype(tk.float32, ("integral", "real floating")))
        self.assertTrue(tk.isdtype(tk.int8, (tk.int8, tk.int16)))
        self.assertFalse(tk.isdtype(tk.int32, ("bool", "real floating")))
        with self.assertRaises(ValueError):
            tk.isdtype(tk.int32, "not a kind")

        # Reachable through the array API namespace.
        xp = tk.array(1.0).__array_namespace__()
        for name in ("result_type", "can_cast", "isdtype", "vecdot"):
            self.assertTrue(hasattr(xp, name), msg=name)


class TestEquality(tiki_tests.TIKITestCase):
    def test_array_eq_array(self):
        a = tk.array([1, 2, 3])
        b = tk.array([1, 2, 3])
        c = tk.array([1, 2, 4])
        self.assertTrue(tk.all(a == b))
        self.assertFalse(tk.all(a == c))

    def test_array_eq_scalar(self):
        a = tk.array([1, 2, 3])
        b = 1
        c = 4
        d = 2.5
        e = tk.array([1, 2.5, 3.25])
        self.assertTrue(tk.any(a == b))
        self.assertFalse(tk.all(a == c))
        self.assertFalse(tk.all(a == d))
        self.assertTrue(tk.any(a == e))

    def test_list_equals_array(self):
        a = tk.array([1, 2, 3])
        b = [1, 2, 3]
        c = [1, 2, 4]

        # tiki array equality returns false if is compared with any kind of
        # object which is not an tiki array
        self.assertFalse(a == b)
        self.assertFalse(a == c)

    def test_tuple_equals_array(self):
        a = tk.array([1, 2, 3])
        b = (1, 2, 3)
        c = (1, 2, 4)

        # tiki array equality returns false if is compared with any kind of
        # object which is not an tiki array
        self.assertFalse(a == b)
        self.assertFalse(a == c)


class TestInequality(tiki_tests.TIKITestCase):
    def test_array_ne_array(self):
        a = tk.array([1, 2, 3])
        b = tk.array([1, 2, 3])
        c = tk.array([1, 2, 4])
        self.assertFalse(tk.any(a != b))
        self.assertTrue(tk.any(a != c))

    def test_array_ne_scalar(self):
        a = tk.array([1, 2, 3])
        b = 1
        c = 4
        d = 1.5
        e = 2.5
        f = tk.array([1, 2.5, 3.25])
        self.assertFalse(tk.all(a != b))
        self.assertTrue(tk.any(a != c))
        self.assertTrue(tk.any(a != d))
        self.assertTrue(tk.any(a != e))
        self.assertFalse(tk.all(a != f))

    def test_list_not_equals_array(self):
        a = tk.array([1, 2, 3])
        b = [1, 2, 3]
        c = [1, 2, 4]

        # tiki array inequality returns true if is compared with any kind of
        # object which is not an tiki array
        self.assertTrue(a != b)
        self.assertTrue(a != c)

    def test_dlx_device_type(self):
        a = tk.array([1, 2, 3])
        device_type, device_id = a.__dlpack_device__()
        self.assertIn(device_type, [1, 8])
        self.assertEqual(device_id, 0)

        if device_type == 8:
            # Additional check if Metal is supposed to be available
            self.assertTrue(tk.metal.is_available())
        elif device_type == 1:
            # Additional check if CPU is the fallback
            self.assertFalse(tk.metal.is_available())

    def test_tuple_not_equals_array(self):
        a = tk.array([1, 2, 3])
        b = (1, 2, 3)
        c = (1, 2, 4)

        # tiki array inequality returns true if is compared with any kind of
        # object which is not an tiki array
        self.assertTrue(a != b)
        self.assertTrue(a != c)

    def test_obj_inequality_array(self):
        str_ = "hello"
        a = tk.array([1, 2, 3])
        lst_ = [1, 2, 3]
        tpl_ = (1, 2, 3)

        # check if object comparison(</>/<=/>=) with tiki array should throw an exception
        # if not, the tests will fail
        with self.assertRaises(ValueError):
            a < str_
        with self.assertRaises(ValueError):
            a > str_
        with self.assertRaises(ValueError):
            a <= str_
        with self.assertRaises(ValueError):
            a >= str_
        with self.assertRaises(ValueError):
            a < lst_
        with self.assertRaises(ValueError):
            a > lst_
        with self.assertRaises(ValueError):
            a <= lst_
        with self.assertRaises(ValueError):
            a >= lst_
        with self.assertRaises(ValueError):
            a < tpl_
        with self.assertRaises(ValueError):
            a > tpl_
        with self.assertRaises(ValueError):
            a <= tpl_
        with self.assertRaises(ValueError):
            a >= tpl_

    def test_invalid_op_on_array(self):
        str_ = "hello"
        a = tk.array([1, 2.5, 3.25])
        lst_ = [1, 2.1, 3.25]
        tpl_ = (1, 2.5, 3.25)

        with self.assertRaises(ValueError):
            a * str_
        with self.assertRaises(ValueError):
            a *= str_
        with self.assertRaises(ValueError):
            a /= lst_
        with self.assertRaises(ValueError):
            a // lst_
        with self.assertRaises(ValueError):
            a % lst_
        with self.assertRaises(ValueError):
            a**tpl_
        with self.assertRaises(ValueError):
            a & tpl_
        with self.assertRaises(ValueError):
            a | str_


class TestArray(tiki_tests.TIKITestCase):
    def test_array_basics(self):
        x = tk.array(1)
        self.assertEqual(x.size, 1)
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.itemsize, 4)
        self.assertEqual(x.nbytes, 4)
        self.assertEqual(x.shape, ())
        self.assertEqual(x.dtype, tk.int32)
        self.assertEqual(x.item(), 1)
        self.assertTrue(isinstance(x.item(), int))

        with self.assertRaises(TypeError):
            len(x)

        x = tk.array(1, tk.uint32)
        self.assertEqual(x.item(), 1)
        self.assertTrue(isinstance(x.item(), int))

        x = tk.array(1, tk.int64)
        self.assertEqual(x.item(), 1)
        self.assertTrue(isinstance(x.item(), int))

        x = tk.array(1, tk.bfloat16)
        self.assertEqual(x.item(), 1.0)

        x = tk.array(1.0)
        self.assertEqual(x.size, 1)
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.shape, ())
        self.assertEqual(x.dtype, tk.float32)
        self.assertEqual(x.item(), 1.0)
        self.assertTrue(isinstance(x.item(), float))

        x = tk.array(False)
        self.assertEqual(x.size, 1)
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.shape, ())
        self.assertEqual(x.dtype, tk.bool_)
        self.assertEqual(x.item(), False)
        self.assertTrue(isinstance(x.item(), bool))

        x = tk.array(complex(1, 1))
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.shape, ())
        self.assertEqual(x.dtype, tk.complex64)
        self.assertEqual(x.item(), complex(1, 1))
        self.assertTrue(isinstance(x.item(), complex))

        x = tk.array([True, False, True])
        self.assertEqual(x.dtype, tk.bool_)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape, (3,))
        self.assertEqual(len(x), 3)

        x = tk.array([True, False, True], tk.float32)
        self.assertEqual(x.dtype, tk.float32)

        x = tk.array([0, 1, 2])
        self.assertEqual(x.dtype, tk.int32)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape, (3,))

        x = tk.array([0, 1, 2], tk.float32)
        self.assertEqual(x.dtype, tk.float32)

        x = tk.array([0.0, 1.0, 2.0])
        self.assertEqual(x.dtype, tk.float32)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape, (3,))

        x = tk.array([1j, 1 + 0j])
        self.assertEqual(x.dtype, tk.complex64)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape, (2,))

        # From tuple
        x = tk.array((1, 2, 3), tk.int32)
        self.assertEqual(x.dtype, tk.int32)
        self.assertEqual(x.tolist(), [1, 2, 3])

    def test_bool_conversion(self):
        x = tk.array(True)
        self.assertTrue(x)
        x = tk.array(False)
        self.assertFalse(x)
        x = tk.array(1.0)
        self.assertTrue(x)
        x = tk.array(0.0)
        self.assertFalse(x)

    def test_int_type(self):
        x = tk.array(1)
        self.assertTrue(x.dtype == tk.int32)
        x = tk.array(2**32 - 1)
        self.assertTrue(x.dtype == tk.int64)
        x = tk.array(2**40)
        self.assertTrue(x.dtype == tk.int64)
        x = tk.array(2**32 - 1, dtype=tk.uint32)
        self.assertTrue(x.dtype == tk.uint32)
        x = tk.array([1, 2], dtype=tk.int64) + 0x80000000
        self.assertTrue(x.dtype == tk.int64)

    def test_construction_from_lists(self):
        x = tk.array([])
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, (0,))
        self.assertEqual(x.dtype, tk.float32)

        x = tk.array([[], [], []])
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, (3, 0))
        self.assertEqual(x.dtype, tk.float32)

        x = tk.array([[[], []], [[], []], [[], []]])
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, (3, 2, 0))
        self.assertEqual(x.dtype, tk.float32)

        # Check failure cases
        with self.assertRaises(ValueError):
            x = tk.array([[[], []], [[]], [[], []]])

        with self.assertRaises(ValueError):
            x = tk.array([[[], []], [[1.0, 2.0], []], [[], []]])

        with self.assertRaises(ValueError):
            x = tk.array([[0, 1], [[0, 1], 1]])

        with self.assertRaises(ValueError):
            x = tk.array([[0, 1], ["hello", 1]])

        x = tk.array([True, False, 3])
        self.assertEqual(x.dtype, tk.int32)

        x = tk.array([True, False, 3, 4.0])
        self.assertEqual(x.dtype, tk.float32)

        x = tk.array([[True, False], [1, 3], [2, 4.0]])
        self.assertEqual(x.dtype, tk.float32)

        x = tk.array([[1.0, 2.0], [0.0, 3.9]], tk.bool_)
        self.assertEqual(x.dtype, tk.bool_)
        self.assertTrue(tk.array_equal(x, tk.array([[True, True], [False, True]])))

        x = tk.array([[1.0, 2.0], [0.0, 3.9]], tk.int32)
        self.assertTrue(tk.array_equal(x, tk.array([[1, 2], [0, 3]])))

        x = tk.array([1 + 0j, 2j, True, 0], tk.complex64)
        self.assertEqual(x.tolist(), [1 + 0j, 2j, 1 + 0j, 0j])

        xnp = np.array([0, 4294967295], dtype=np.uint32)
        x = tk.array([0, 4294967295], dtype=tk.uint32)
        self.assertTrue(np.array_equal(x, xnp))

        xnp = np.array([0, 4294967295], dtype=np.float32)
        x = tk.array([0, 4294967295], dtype=tk.float32)
        self.assertTrue(np.array_equal(x, xnp))

    def test_double_keeps_precision(self):
        x = 39.14223403241
        out = tk.array(x, dtype=tk.float64).item()
        self.assertEqual(out, x)

        out = tk.array([x], dtype=tk.float64).item()
        self.assertEqual(out, x)

    def test_construction_from_lists_wide_ints(self):
        # A python int that does not fit in int32 widens to int64, the same
        # rule the scalar path already uses. It used to raise std::bad_cast.
        for value in (2**31, 2**40, -(2**31) - 1, -(2**40)):
            for make in (
                lambda v: [v],
                lambda v: (v,),
                lambda v: [[v]],
                lambda v: [v, 1],
            ):
                x = tk.array(make(value))
                self.assertEqual(x.dtype, tk.int64, msg=f"{value} {make(value)}")
                self.assertEqual(x.flatten()[0].item(), value)
                self.assertEqual(tk.array(value).dtype, tk.int64)

        # Values that still fit keep int32, including both boundaries.
        for value in (0, 1, 2**31 - 1, -(2**31)):
            x = tk.array([value])
            self.assertEqual(x.dtype, tk.int32, msg=str(value))
            self.assertEqual(x[0].item(), value)

        # An explicit dtype still wins.
        self.assertEqual(tk.array([2**40], tk.int64).dtype, tk.int64)
        self.assertEqual(tk.array([1, 2], tk.int64).dtype, tk.int64)
        # A float in the list still makes it float, not int64.
        self.assertEqual(tk.array([2**40, 1.5]).dtype, tk.float32)

    def test_construction_from_lists_of_tiki_arrays(self):
        dtypes = [
            tk.bool_,
            tk.uint8,
            tk.uint16,
            tk.uint32,
            tk.uint64,
            tk.int8,
            tk.int16,
            tk.int32,
            tk.int64,
            tk.float16,
            tk.float32,
            tk.bfloat16,
            tk.complex64,
        ]
        for x_t, y_t in permutations(dtypes, 2):
            # check type promotion and numeric correctness
            x, y = tk.array([1.0], x_t), tk.array([2.0], y_t)
            z = tk.array([x, y])
            expected = tk.stack([x, y], axis=0)
            self.assertEqualArray(z, expected)

            # check heterogeneous construction with tiki arrays and python primitive types
            x, y = tk.array([True], x_t), tk.array([False], y_t)
            z = tk.array([[x, [2.0]], [[3.0], y]])
            expected = tk.array([[[x.item()], [2.0]], [[3.0], [y.item()]]], z.dtype)
            self.assertEqualArray(z, expected)

        # check when create from an array which does not contain memory to the raw data
        x = tk.array([1.0]).astype(tk.bfloat16)  # x does not hold raw data
        for y_t in dtypes:
            y = tk.array([2.0], y_t)
            z = tk.array([x, y])
            expected = tk.stack([x, y], axis=0)
            self.assertEqualArray(z, expected)

        # shape check from `stack()`
        with self.assertRaises(ValueError) as e:
            tk.array([x, 1.0])
        self.assertEqual(
            str(e.exception), "Initialization encountered non-uniform length."
        )

        # shape check from `validate_shape`
        with self.assertRaises(ValueError) as e:
            tk.array([1.0, x])
        self.assertEqual(
            str(e.exception), "Initialization encountered non-uniform length."
        )

        # check that `[tk.array, ...]` retains the `tk.array` in the graph
        def f(x):
            y = tk.array([x, tk.array([2.0])])
            return (2 * y).sum()

        x = tk.array([1.0])
        dfdx = tk.grad(f)
        self.assertEqual(dfdx(x).item(), 2.0)

    def test_init_from_array(self):
        x = tk.array(3.0)
        y = tk.array(x)

        self.assertTrue(tk.array_equal(x, y))

        y = tk.array(x, tk.int32)
        self.assertEqual(y.dtype, tk.int32)
        self.assertEqual(y.item(), 3)

        y = tk.array(x, tk.bool_)
        self.assertEqual(y.dtype, tk.bool_)
        self.assertEqual(y.item(), True)

        y = tk.array(x, tk.complex64)
        self.assertEqual(y.dtype, tk.complex64)
        self.assertEqual(y.item(), 3.0 + 0j)

    def test_array_repr(self):
        x = tk.array(True)
        self.assertEqual(str(x), "array(True, dtype=bool)")
        x = tk.array(1)
        self.assertEqual(str(x), "array(1, dtype=int32)")
        x = tk.array(1.0)
        self.assertEqual(str(x), "array(1, dtype=float32)")

        x = tk.array([1, 0, 1])
        self.assertEqual(str(x), "array([1, 0, 1], dtype=int32)")

        x = tk.array([1] * 6)
        expected = "array([1, 1, 1, 1, 1, 1], dtype=int32)"
        self.assertEqual(str(x), expected)

        x = tk.array([1] * 7)
        expected = "array([1, 1, 1, ..., 1, 1, 1], dtype=int32)"
        self.assertEqual(str(x), expected)

        x = tk.array([[1, 2], [1, 2], [1, 2]])
        expected = "array([[1, 2],\n       [1, 2],\n       [1, 2]], dtype=int32)"
        self.assertEqual(str(x), expected)

        x = tk.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
        expected = (
            "array([[[1, 2],\n"
            "        [1, 2]],\n"
            "       [[1, 2],\n"
            "        [1, 2]]], dtype=int32)"
        )
        self.assertEqual(str(x), expected)

        x = tk.array([[1, 2]] * 6)
        expected = (
            "array([[1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2]], dtype=int32)"
        )
        self.assertEqual(str(x), expected)
        x = tk.array([[1, 2]] * 7)
        expected = (
            "array([[1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       ...,\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2]], dtype=int32)"
        )
        self.assertEqual(str(x), expected)

        x = tk.array([1], dtype=tk.int8)
        expected = "array([1], dtype=int8)"
        self.assertEqual(str(x), expected)
        x = tk.array([1], dtype=tk.int16)
        expected = "array([1], dtype=int16)"
        self.assertEqual(str(x), expected)
        x = tk.array([1], dtype=tk.uint8)
        expected = "array([1], dtype=uint8)"
        self.assertEqual(str(x), expected)

        # Fp16 is not supported in all platforms
        x = tk.array([1.2], dtype=tk.float16)
        expected = "array([1.2002], dtype=float16)"
        self.assertEqual(str(x), expected)

        x = tk.array([1 + 1j], dtype=tk.complex64)
        expected = "array([1+1j], dtype=complex64)"
        self.assertEqual(str(x), expected)
        x = tk.array([1 - 1j], dtype=tk.complex64)
        expected = "array([1-1j], dtype=complex64)"

        x = tk.array([1 + 1j], dtype=tk.complex64)
        expected = "array([1+1j], dtype=complex64)"
        self.assertEqual(str(x), expected)
        x = tk.array([1 - 1j], dtype=tk.complex64)
        expected = "array([1-1j], dtype=complex64)"

    def test_array_repr_precision(self):
        x = tk.array([1.123456789], dtype=tk.float32)
        expected = "array([1.12346], dtype=float32)"
        self.assertEqual(str(x), expected)

        with tk.printoptions(precision=4):
            expected = "array([1.1235], dtype=float32)"
            self.assertEqual(str(x), expected)
        tk.set_printoptions(precision=2)
        expected = "array([1.12], dtype=float32)"
        self.assertEqual(str(x), expected)

        x = tk.sin(x)
        expected = "array([0.90], dtype=float32)"
        self.assertEqual(str(x), expected)

        with tk.printoptions(precision=4):
            expected = "array([0.9016], dtype=float32)"
            self.assertEqual(str(x), expected)

    def test_array_to_list(self):
        types = [tk.bool_, tk.uint32, tk.int32, tk.int64, tk.float32]
        for t in types:
            x = tk.array(1, t)
            self.assertEqual(x.tolist(), 1)

        vals = [1, 2, 3, 4]
        x = tk.array(vals)
        self.assertEqual(x.tolist(), vals)

        vals = [[1, 2], [3, 4]]
        x = tk.array(vals)
        self.assertEqual(x.tolist(), vals)

        vals = [[1, 0], [0, 1]]
        x = tk.array(vals, tk.bool_)
        self.assertEqual(x.tolist(), vals)

        vals = [[1.5, 2.5], [3.5, 4.5]]
        x = tk.array(vals)
        self.assertEqual(x.tolist(), vals)

        vals = [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]]
        x = tk.array(vals)
        self.assertEqual(x.tolist(), vals)

        # Empty arrays
        vals = []
        x = tk.array(vals)
        self.assertEqual(x.tolist(), vals)

        vals = [[], []]
        x = tk.array(vals)
        self.assertEqual(x.tolist(), vals)

        # Complex arrays
        vals = [0.5 + 0j, 1.5 + 1j, 2.5 + 0j, 3.5 + 1j]
        x = tk.array(vals)
        self.assertEqual(x.tolist(), vals)

        # Half types
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = tk.array(vals, dtype=tk.float16)
        self.assertEqual(x.tolist(), vals)

        x = tk.array(vals, dtype=tk.bfloat16)
        self.assertEqual(x.tolist(), vals)

    def test_array_np_conversion(self):
        # Shape test
        a = np.array([])
        x = tk.array(a)
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, (0,))
        self.assertEqual(x.dtype, tk.float32)

        a = np.array([[], [], []])
        x = tk.array(a)
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, (3, 0))
        self.assertEqual(x.dtype, tk.float32)

        a = np.array([[[], []], [[], []], [[], []]])
        x = tk.array(a)
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, (3, 2, 0))
        self.assertEqual(x.dtype, tk.float32)

        # Content test
        a = 2.0 * np.ones((3, 5, 4))
        x = tk.array(a)
        self.assertEqual(x.dtype, tk.float32)
        self.assertEqual(x.ndim, 3)
        self.assertEqual(x.shape, (3, 5, 4))

        y = np.asarray(x)
        self.assertTrue(np.allclose(a, y))

        a = np.array(3, dtype=np.int32)
        x = tk.array(a)
        self.assertEqual(x.dtype, tk.int32)
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.shape, ())
        self.assertEqual(x.item(), 3)

        # tiki to numpy test
        x = tk.array([True, False, True])
        y = np.asarray(x)
        self.assertEqual(y.dtype, np.bool_)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y[0], True)
        self.assertEqual(y[1], False)
        self.assertEqual(y[2], True)

        # complex64 tk <-> np
        cvals = [0j, 1, 1 + 1j]
        x = np.array(cvals)
        y = tk.array(x)
        self.assertEqual(y.dtype, tk.complex64)
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.tolist(), cvals)

        y = tk.array([0j, 1, 1 + 1j])
        x = np.asarray(y)
        self.assertEqual(x.dtype, np.complex64)
        self.assertEqual(x.shape, (3,))
        self.assertEqual(x.tolist(), cvals)

    @unittest.skipUnless(tk.cuda.is_available(), "requires CUDA")
    def test_numpy_export_waits_for_device_copy(self) -> None:
        # Compute Sanitizer checks that host export finishes before device free.
        for size in (5, 262144):
            with self.subTest(size=size), tk.stream(tk.gpu):
                x = tk.arange(size, dtype=tk.float32) + 1
                tk.eval(x)
                # Isolate the host copy from event-query tracking in the sanitizer.
                tk.synchronize(tk.default_stream(tk.gpu))
                np.testing.assert_array_equal(
                    np.asarray(x), np.arange(1, size + 1, dtype=np.float32)
                )

    def test_array_np_dtype_conversion(self):
        dtypes_list = [
            (tk.bool_, np.bool_),
            (tk.uint8, np.uint8),
            (tk.uint16, np.uint16),
            (tk.uint32, np.uint32),
            (tk.uint64, np.uint64),
            (tk.int8, np.int8),
            (tk.int16, np.int16),
            (tk.int32, np.int32),
            (tk.int64, np.int64),
            (tk.float16, np.float16),
            (tk.float32, np.float32),
            (tk.complex64, np.complex64),
        ]

        for tiki_dtype, np_dtype in dtypes_list:
            a_npy = np.random.uniform(low=0, high=100, size=(32,)).astype(np_dtype)
            a_tiki = tk.array(a_npy)

            self.assertEqual(a_tiki.dtype, tiki_dtype)
            self.assertTrue(np.allclose(a_tiki, a_npy))

            b_tiki = tk.random.uniform(
                low=0,
                high=10,
                shape=(32,),
            ).astype(tiki_dtype)
            b_npy = np.array(b_tiki)

            self.assertEqual(b_npy.dtype, np_dtype)

    def test_array_from_noncontiguous_np(self):
        for t in [np.int8, np.int32, np.float16, np.float32, np.complex64]:
            np_arr = np.random.uniform(size=(10, 10)).astype(np.complex64)
            np_arr = np_arr.T
            mx_arr = tk.array(np_arr)
            self.assertTrue(tk.array_equal(np_arr, mx_arr))

    def test_array_np_shape_dim_check(self):
        a_npy = np.empty(2**31, dtype=np.bool_)
        with self.assertRaises(OverflowError) as e:
            tk.array(a_npy)
        self.assertEqual(
            str(e.exception),
            "Shape dimension 2147483648 is outside the supported range "
            "[-2147483648, 2147483647]. Tiki currently uses 32-bit integers "
            "for shape dimensions.",
        )

    def test_dtype_promotion(self):
        dtypes_list = [
            (tk.bool_, np.bool_),
            (tk.uint8, np.uint8),
            (tk.uint16, np.uint16),
            (tk.uint32, np.uint32),
            (tk.uint64, np.uint64),
            (tk.int8, np.int8),
            (tk.int16, np.int16),
            (tk.int32, np.int32),
            (tk.int64, np.int64),
            (tk.float32, np.float32),
        ]

        promotion_pairs = permutations(dtypes_list, 2)

        for (tiki_dt_1, np_dt_1), (tiki_dt_2, np_dt_2) in promotion_pairs:
            with self.subTest(dtype1=np_dt_1, dtype2=np_dt_2):
                a_npy = np.ones((3,), dtype=np_dt_1)
                b_npy = np.ones((3,), dtype=np_dt_2)

                c_npy = a_npy + b_npy

                a_tiki = tk.ones((3,), dtype=tiki_dt_1)
                b_tiki = tk.ones((3,), dtype=tiki_dt_2)

                c_tiki = a_tiki + b_tiki

                self.assertEqual(c_tiki.dtype, tk.array(c_npy).dtype)

        a_tiki = tk.ones((3,), dtype=tk.float16)
        b_tiki = tk.ones((3,), dtype=tk.float32)
        c_tiki = a_tiki + b_tiki

        self.assertEqual(c_tiki.dtype, tk.float32)

        b_tiki = tk.ones((3,), dtype=tk.int32)
        c_tiki = a_tiki + b_tiki

        self.assertEqual(c_tiki.dtype, tk.float16)

    def test_dtype_python_scalar_promotion(self):
        tests = [
            (tk.bool_, operator.mul, False, tk.bool_),
            (tk.bool_, operator.mul, 0, tk.int32),
            (tk.bool_, operator.mul, 1.0, tk.float32),
            (tk.int8, operator.mul, False, tk.int8),
            (tk.int8, operator.mul, 0, tk.int8),
            (tk.int8, operator.mul, 1.0, tk.float32),
            (tk.int16, operator.mul, False, tk.int16),
            (tk.int16, operator.mul, 0, tk.int16),
            (tk.int16, operator.mul, 1.0, tk.float32),
            (tk.int32, operator.mul, False, tk.int32),
            (tk.int32, operator.mul, 0, tk.int32),
            (tk.int32, operator.mul, 1.0, tk.float32),
            (tk.int64, operator.mul, False, tk.int64),
            (tk.int64, operator.mul, 0, tk.int64),
            (tk.int64, operator.mul, 1.0, tk.float32),
            (tk.uint8, operator.mul, False, tk.uint8),
            (tk.uint8, operator.mul, 0, tk.uint8),
            (tk.uint8, operator.mul, 1.0, tk.float32),
            (tk.uint16, operator.mul, False, tk.uint16),
            (tk.uint16, operator.mul, 0, tk.uint16),
            (tk.uint16, operator.mul, 1.0, tk.float32),
            (tk.uint32, operator.mul, False, tk.uint32),
            (tk.uint32, operator.mul, 0, tk.uint32),
            (tk.uint32, operator.mul, 1.0, tk.float32),
            (tk.uint64, operator.mul, False, tk.uint64),
            (tk.uint64, operator.mul, 0, tk.uint64),
            (tk.uint64, operator.mul, 1.0, tk.float32),
            (tk.float32, operator.mul, False, tk.float32),
            (tk.float32, operator.mul, 0, tk.float32),
            (tk.float32, operator.mul, 1.0, tk.float32),
            (tk.float16, operator.mul, False, tk.float16),
            (tk.float16, operator.mul, 0, tk.float16),
            (tk.float16, operator.mul, 1.0, tk.float16),
        ]

        for dtype_in, f, v, dtype_out in tests:
            x = tk.array(0, dtype_in)
            y = f(x, v)
            self.assertEqual(y.dtype, dtype_out)

    def test_array_comparison(self):
        a = tk.array([0.0, 1.0, 5.0])
        b = tk.array([-1.0, 2.0, 5.0])

        self.assertEqual((a < b).tolist(), [False, True, False])
        self.assertEqual((a <= b).tolist(), [False, True, True])
        self.assertEqual((a > b).tolist(), [True, False, False])
        self.assertEqual((a >= b).tolist(), [True, False, True])

        self.assertEqual((a < 5).tolist(), [True, True, False])
        self.assertEqual((5 < a).tolist(), [False, False, False])
        self.assertEqual((5 <= a).tolist(), [False, False, True])
        self.assertEqual((a > 1).tolist(), [False, False, True])
        self.assertEqual((a >= 1).tolist(), [False, True, True])

    def test_array_neg(self):
        a = tk.array([-1.0, 4.0, 0.0])

        self.assertEqual((-a).tolist(), [1.0, -4.0, 0.0])

    def test_array_type_cast(self):
        a = tk.array([0.1, 2.3, -1.3])
        b = [0, 2, -1]

        self.assertEqual(a.astype(tk.int32).tolist(), b)
        self.assertEqual(a.astype(tk.int32).dtype, tk.int32)

        b = tk.array(b).astype(tk.float32)
        self.assertEqual(b.dtype, tk.float32)

    def test_array_iteration(self):
        a = tk.array([0, 1, 2])

        for i, x in enumerate(a):
            self.assertEqual(x.item(), i)

        a = tk.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x, y, z = a
        self.assertEqual(x.tolist(), [1.0, 2.0])
        self.assertEqual(y.tolist(), [3.0, 4.0])
        self.assertEqual(z.tolist(), [5.0, 6.0])

        a = tk.array(3)
        with self.assertRaises(TypeError):
            list(a)

    def test_array_pickle(self):
        dtypes = [
            tk.int8,
            tk.int16,
            tk.int32,
            tk.int64,
            tk.uint8,
            tk.uint16,
            tk.uint32,
            tk.uint64,
            tk.float16,
            tk.float32,
            tk.bfloat16,
            tk.complex64,
        ]

        for dtype in dtypes:
            x = tk.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype)
            state = pickle.dumps(x)
            y = pickle.loads(state)
            self.assertEqualArray(y, x)

    def test_array_copy(self):
        dtypes = [
            tk.int8,
            tk.int16,
            tk.int32,
            tk.int64,
            tk.uint8,
            tk.uint16,
            tk.uint32,
            tk.uint64,
            tk.float16,
            tk.float32,
            tk.bfloat16,
            tk.complex64,
        ]

        for copy_function in [copy, deepcopy]:
            for dtype in dtypes:
                x = tk.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype)
                y = copy_function(x)
                self.assertEqualArray(y, x)

                y -= 1
                self.assertEqualArray(y, x - 1)

    def test_indexing(self):
        # Only ellipsis is a no-op
        a_tiki = tk.array([1])[...]
        self.assertEqual(a_tiki.shape, (1,))
        self.assertEqual(a_tiki.item(), 1)

        # Basic content check, slice indexing
        a_npy = np.arange(64, dtype=np.float32)
        a_tiki = tk.array(a_npy)
        a_sliced_tiki = a_tiki[2:50:4]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[2:50:4]))

        # Basic content check, tiki array indexing
        a_npy = np.arange(64, dtype=np.int32)
        a_npy = a_npy.reshape((8, 8))
        a_tiki = tk.array(a_npy)
        idx_npy = np.array([0, 1, 2, 7, 5], dtype=np.uint32)
        idx_tiki = tk.array(idx_npy)
        a_sliced_tiki = a_tiki[idx_tiki]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[idx_npy]))

        # Basic content check, int indexing
        a_sliced_tiki = a_tiki[5]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[5]))
        self.assertEqual(len(a_sliced_npy.shape), len(a_npy[5].shape))
        self.assertEqual(len(a_sliced_npy.shape), 1)
        self.assertEqual(a_sliced_npy.shape[0], a_npy[5].shape[0])

        # Basic content check, negative indexing
        a_sliced_tiki = a_tiki[-1]
        self.assertTrue(np.array_equal(a_sliced_tiki, a_npy[-1]))

        # NumPy integer scalar indexing
        a_sliced_tiki = a_tiki[np.int64(5)]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[np.int64(5)]))

        # Basic content check, empty index
        a_sliced_tiki = a_tiki[()]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[()]))

        # Basic content check, new axis
        a_sliced_tiki = a_tiki[None]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[None]))

        a_sliced_tiki = a_tiki[:, None]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[:, None]))

        # Multi dim indexing, all ints
        self.assertEqual(a_tiki[0, 0].item(), 0)
        self.assertEqual(a_tiki[0, 0].ndim, 0)

        # Multi dim indexing, all slices
        a_sliced_tiki = a_tiki[2:4, 5:]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[2:4, 5:]))

        a_sliced_tiki = a_tiki[:, 0:5]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[:, 0:5]))

        # Slicing, strides
        a_sliced_tiki = a_tiki[:, ::2]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[:, ::2]))

        # Slicing, -ve index
        a_sliced_tiki = a_tiki[-2:, :-1]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[-2:, :-1]))

        # Slicing, start > end
        a_sliced_tiki = a_tiki[8:3]
        self.assertEqual(a_sliced_tiki.size, 0)

        # Slicing, Clipping past the end
        a_sliced_tiki = a_tiki[7:10]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[7:10]))

        # Multi dim indexing, int and slices
        a_sliced_tiki = a_tiki[0, :5]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[0, :5]))

        a_sliced_tiki = a_tiki[:, -1]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[:, -1]))

        # Multi dim indexing, int and array
        a_sliced_tiki = a_tiki[idx_tiki, 0]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[idx_npy, 0]))

        # Multi dim indexing, array and slices
        a_sliced_tiki = a_tiki[idx_tiki, :5]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[idx_npy, :5]))

        a_sliced_tiki = a_tiki[:, idx_tiki]
        a_sliced_npy = np.asarray(a_sliced_tiki)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[:, idx_npy]))

        # Multi dim indexing with multiple arrays
        def check_slices(arr_np, *idx_np):
            arr_tiki = tk.array(arr_np)
            idx_tiki = [
                tk.array(idx) if isinstance(idx, np.ndarray) else idx for idx in idx_np
            ]
            slice_tiki = arr_tiki[tuple(idx_tiki)]
            self.assertTrue(
                np.array_equal(arr_np[tuple(idx_np)], arr_tiki[tuple(idx_tiki)])
            )

        a_np = np.arange(16).reshape(4, 4)
        check_slices(a_np, np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
        check_slices(a_np, np.array([0, 1, 2, 3]), np.array([1, 0, 3, 3]))
        check_slices(a_np, np.array([[0, 1]]), np.array([[0], [1], [3]]))

        a_np = np.arange(64).reshape(2, 4, 2, 4)
        check_slices(a_np, 0, np.array([0, 1, 2]))
        check_slices(a_np, slice(0, 1), np.array([0, 1, 2]))
        check_slices(
            a_np, slice(0, 1), np.array([0, 1, 2]), slice(None), slice(0, 4, 2)
        )
        check_slices(
            a_np, slice(0, 1), np.array([0, 1, 2]), slice(None), np.array([1, 2, 0])
        )
        check_slices(a_np, slice(0, 1), np.array([0, 1, 2]), 1, np.array([1, 2, 0]))
        check_slices(
            a_np, slice(0, 1), np.array([0, 1, 2]), np.array([1, 0, 0]), slice(0, 1)
        )
        check_slices(
            a_np,
            slice(0, 1),
            np.array([[0], [1], [2]]),
            np.array([[1, 0, 0]]),
            slice(0, 1),
        )
        check_slices(
            a_np,
            slice(0, 2),
            np.array([[0], [1], [2]]),
            slice(0, 2),
            np.array([[1, 0, 0]]),
        )
        for p in permutations([slice(None), slice(None), 0, np.array([1, 0])]):
            check_slices(a_np, *p)
        for p in permutations(
            [slice(None), slice(None), 0, np.array([1, 0]), None, None]
        ):
            check_slices(a_np, *p)
        for p in permutations([0, np.array([1, 0]), None, Ellipsis, slice(None)]):
            check_slices(a_np, *p)

        # Non-contiguous arrays in slicing
        a_tiki = tk.reshape(tk.arange(128), (16, 8))
        a_tiki = a_tiki[::2, :]
        a_np = np.array(a_tiki)
        idx_np = np.arange(8)[::2]
        idx_tiki = tk.arange(8)[::2]
        self.assertTrue(
            np.array_equal(a_np[idx_np, idx_np], np.array(a_tiki[idx_tiki, idx_tiki]))
        )

        # Slicing with negative indices and integer
        a_np = np.arange(10).reshape(5, 2)
        a_tiki = tk.array(a_np)
        self.assertTrue(np.array_equal(a_np[2:-1, 0], np.array(a_tiki[2:-1, 0])))

        # Ellipsis with more trailing indices than dimensions
        a_tiki = tk.array([1, 2, 3])
        with self.assertRaises(ValueError):
            a_tiki[..., 0, 0]
        with self.assertRaises(ValueError):
            a_tiki[..., 0, 0] = 5

    def test_indexing_grad(self):
        x = tk.array([[1, 2], [3, 4]]).astype(tk.float32)
        ind = tk.array([0, 1, 0]).astype(tk.float32)

        def index_fn(x, ind):
            return x[tk.stop_gradient(ind.astype(tk.int32))].sum()

        grad_x, grad_ind = tk.grad(index_fn, argnums=(0, 1))(x, ind)
        expected = tk.array([[2, 2], [1, 1]])

        self.assertTrue(tk.array_equal(grad_x, expected))
        self.assertTrue(tk.array_equal(grad_ind, tk.zeros(ind.shape)))

    def test_setitem(self):
        a = tk.array(0)
        a[None] = 1
        self.assertEqual(a.item(), 1)

        a = tk.array([1, 2, 3])
        a[0] = 2
        self.assertEqual(a.tolist(), [2, 2, 3])

        a[-1] = 2
        self.assertEqual(a.tolist(), [2, 2, 2])

        a[np.int64(1)] = 9
        self.assertEqual(a.tolist(), [2, 9, 2])

        a[0] = tk.array([[[1]]])
        self.assertEqual(a.tolist(), [1, 9, 2])

        a[:] = 0
        self.assertEqual(a.tolist(), [0, 0, 0])

        a[None] = 1
        self.assertEqual(a.tolist(), [1, 1, 1])

        a[0:1] = 2
        self.assertEqual(a.tolist(), [2, 1, 1])

        a[0:2] = 3
        self.assertEqual(a.tolist(), [3, 3, 1])

        # Assigning through a bare Ellipsis, like a[:] and a[None]
        e = tk.zeros((2, 3), tk.int32)
        e[...] = 5
        self.assertEqual(e.tolist(), [[5, 5, 5], [5, 5, 5]])

        # Broadcasting an array update through Ellipsis
        e[...] = tk.array([1, 2, 3])
        self.assertEqual(e.tolist(), [[1, 2, 3], [1, 2, 3]])

        e[...] = tk.zeros((2, 3), tk.int32)
        self.assertEqual(e.tolist(), [[0, 0, 0], [0, 0, 0]])

        # Scalar array
        e = tk.array(0)
        e[...] = 7
        self.assertEqual(e.item(), 7)

        # Shapes that cannot broadcast are still rejected
        e = tk.zeros((2, 3), tk.int32)
        with self.assertRaises(ValueError):
            e[...] = tk.array([1, 2])

        a[0:3] = 4
        self.assertEqual(a.tolist(), [4, 4, 4])

        a[0:1] = tk.array(0)
        self.assertEqual(a.tolist(), [0, 4, 4])

        a[0:1] = tk.array([1])
        self.assertEqual(a.tolist(), [1, 4, 4])

        # Regression test: a negative integer index after a None
        # (newaxis) used to be normalized against the wrong axis size,
        # silently writing nothing instead of updating the last row.
        b = tk.zeros((3, 4))
        b[None, -1] = 9
        self.assertEqual(b.tolist(), [[0, 0, 0, 0], [0, 0, 0, 0], [9, 9, 9, 9]])

        with self.assertRaises(ValueError):
            a[0:1] = tk.array([2, 3])

        a[0:2] = tk.array([2, 2])
        self.assertEqual(a.tolist(), [2, 2, 4])

        a[:] = tk.array([[[[1, 1, 1]]]])
        self.assertEqual(a.tolist(), [1, 1, 1])

        # Array slices
        def check_slices(arr_np, update_np, *idx_np):
            arr_tiki = tk.array(arr_np)
            update_tiki = tk.array(update_np)
            idx_tiki = [
                tk.array(idx) if isinstance(idx, np.ndarray) else idx for idx in idx_np
            ]
            if len(idx_np) > 1:
                idx_np = tuple(idx_np)
                idx_tiki = tuple(idx_tiki)
            else:
                idx_np = idx_np[0]
                idx_tiki = idx_tiki[0]
            arr_np[idx_np] = update_np
            arr_tiki[idx_tiki] = update_tiki
            self.assertTrue(np.array_equal(arr_np, arr_tiki))

        check_slices(np.zeros((3, 3)), 1, 0)
        check_slices(np.zeros((3, 3)), 1, -1)
        check_slices(np.zeros((3, 3)), 1, slice(0, 2))
        check_slices(np.zeros((3, 3)), np.array([[0, 1, 2], [3, 4, 5]]), slice(0, 2))

        with self.assertRaises(ValueError):
            a = tk.array(0)
            a[0] = tk.array(1)

        check_slices(np.zeros((3, 3)), 1, np.array([0, 1, 2]))
        check_slices(np.zeros((3, 3)), np.array(3), np.array([0, 1, 2]))
        check_slices(np.zeros((3, 3)), np.array([3]), np.array([0, 1, 2]))
        check_slices(np.zeros((3, 3)), np.array([3]), np.array([0, 1]))
        check_slices(np.zeros((3, 2)), np.array([[3, 3], [4, 4]]), np.array([0, 1]))
        check_slices(np.zeros((3, 2)), np.array([[3, 3], [4, 4]]), np.array([0, 1]))
        check_slices(
            np.zeros((3, 2)), np.array([[3, 3], [4, 4], [5, 5]]), np.array([0, 2, 1])
        )

        # Multiple slices
        a = tk.array(0)
        a[None, None] = 1
        self.assertEqual(a.item(), 1)

        a[None, None] = tk.array(2)
        self.assertEqual(a.item(), 2)

        a[None, None] = tk.array([[[3]]])
        self.assertEqual(a.item(), 3)

        a[()] = 4
        self.assertEqual(a.item(), 4)

        a_np = np.zeros((2, 3, 4, 5))
        check_slices(a_np, 1, np.array([0, 0]), slice(0, 2), slice(0, 3), 4)
        check_slices(
            a_np,
            np.arange(10).reshape(2, 5),
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([2, 3]),
        )
        check_slices(
            a_np,
            np.array([[3], [4]]),
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([2, 3]),
        )
        check_slices(
            a_np, np.arange(5), np.array([0, 0]), np.array([0, 1]), np.array([2, 3])
        )
        check_slices(np.zeros(5), np.arange(2), None, None, np.array([2, 3]))
        check_slices(
            np.zeros((4, 3, 4)),
            np.arange(3),
            np.array([2, 3]),
            slice(0, 3),
            np.array([2, 3]),
        )

        with self.assertRaises(ValueError):
            a = tk.zeros((4, 3, 4))
            a[tk.array([2, 3]), None, tk.array([2, 3])] = tk.arange(2)

        with self.assertRaises(ValueError):
            a = tk.zeros((4, 3, 4))
            a[tk.array([2, 3]), None, tk.array([2, 3])] = tk.arange(3)

        check_slices(np.zeros((4, 3, 4)), 1, np.array([2, 3]), None, np.array([2, 1]))
        check_slices(
            np.zeros((4, 3, 4)), np.arange(4), np.array([2, 3]), None, np.array([2, 1])
        )
        check_slices(
            np.zeros((4, 3, 4)),
            np.arange(2 * 4).reshape(2, 1, 4),
            np.array([2, 3]),
            None,
            np.array([2, 1]),
        )

        check_slices(np.zeros((4, 4)), 1, slice(0, 2), slice(0, 2))
        check_slices(np.zeros((4, 4)), np.arange(2), slice(0, 2), slice(0, 2))
        check_slices(
            np.zeros((4, 4)), np.arange(2).reshape(2, 1), slice(0, 2), slice(0, 2)
        )
        check_slices(
            np.zeros((4, 4)), np.arange(4).reshape(2, 2), slice(0, 2), slice(0, 2)
        )

        with self.assertRaises(ValueError):
            a = tk.zeros((2, 2, 2))
            a[..., ...] = 1

        with self.assertRaises(ValueError):
            a = tk.zeros((2, 2, 2, 2, 2))
            a[0, ..., 0, ..., 0] = 1

        with self.assertRaises(ValueError):
            a = tk.zeros((2, 2))
            a[0, 0, 0] = 1

        with self.assertRaises(ValueError):
            a = tk.zeros((5, 4, 3))
            a[:, 0] = tk.ones((5, 1, 3))

        check_slices(np.zeros((2, 2, 2, 2)), 1, None, Ellipsis, None)
        check_slices(
            np.zeros((2, 2, 2, 2)), 1, np.array([0, 1]), Ellipsis, np.array([0, 1])
        )
        check_slices(
            np.zeros((2, 2, 2, 2)),
            np.arange(2 * 2 * 2).reshape(2, 2, 2),
            np.array([0, 1]),
            Ellipsis,
            np.array([0, 1]),
        )

        # Check slice assign with negative indices works
        a = tk.zeros((5, 5), tk.int32)
        a[2:-2, 2:-2] = 4
        self.assertEqual(a[2, 2].item(), 4)

        # Check slice array slice
        check_slices(
            np.zeros((5, 4, 4)),
            np.arange(4 * 2 * 3).reshape(4, 2, 3),
            slice(0, 4),
            np.array([1, 3]),
            slice(None, -1),
        )
        check_slices(
            np.zeros((5, 4, 4)),
            np.arange(4 * 2 * 2).reshape(4, 2, 2),
            slice(0, 4),
            np.array([1, 3]),
            slice(0, 4, 2),
        )

        check_slices(
            np.zeros((1, 10, 4)),
            np.arange(2 * 4).reshape(1, 2, 4),
            slice(None, None, None),
            np.array([1, 3]),
        )

        check_slices(
            np.zeros((3, 4, 5, 3)),
            np.arange(2 * 4 * 3 * 3).reshape(2, 4, 3, 3),
            np.array([2, 1]),
            slice(None, None, None),
            slice(None, None, 2),
            slice(None, None, None),
        )

        check_slices(
            np.zeros((3, 4, 5, 3)),
            np.arange(2 * 4 * 3 * 3).reshape(2, 4, 3, 3),
            np.array([2, 1]),
            slice(None, None, None),
            slice(None, None, 2),
        )

        check_slices(np.zeros((5, 4, 3)), np.ones((5, 3)), slice(None), 0)

        check_slices(np.zeros((5, 4, 3)), np.ones((5, 1, 3)), slice(None), slice(0, 1))
        check_slices(
            np.ones((3, 4, 4, 4)), np.zeros((4, 4)), 0, slice(0, 4), 3, slice(0, 4)
        )

        x = tk.zeros((2, 3, 4, 5, 3))
        x[..., 0] = 1.0
        self.assertTrue(tk.array_equal(x[..., 0], tk.ones((2, 3, 4, 5))))

        x = tk.zeros((2, 3, 4, 5, 3))
        x[:, 0] = 1.0
        self.assertTrue(tk.array_equal(x[:, 0], tk.ones((2, 4, 5, 3))))

        x = tk.zeros((2, 2, 2, 2, 2, 2))
        x[0, 0] = 1
        self.assertTrue(tk.array_equal(x[0, 0], tk.ones((2, 2, 2, 2))))

        a = tk.zeros((2, 2, 2))
        with self.assertRaises(ValueError):
            a[:, None, :] = tk.ones((2, 2, 2))

        # Ok, doesn't throw
        a[:, None, :] = tk.ones((2, 1, 2, 2))
        a[:, None, :] = tk.ones((2, 2))
        a[:, None, 0] = tk.ones((2,))
        a[:, None, 0] = tk.ones((1, 2))

    def test_array_at(self):
        a = tk.array(1)
        with self.assertRaises(ValueError):
            a.at.add(1)

        a = a.at[None].add(1)
        self.assertEqual(a.item(), 2)

        a = tk.array([0, 1, 2])
        a = a.at[1].add(2)
        self.assertEqual(a.tolist(), [0, 3, 2])

        a = a.at[tk.array([0, 0, 0, 0])].add(1)
        self.assertEqual(a.tolist(), [4, 3, 2])

        a = tk.zeros((10, 10))
        a = a.at[0].add(tk.arange(10))
        self.assertEqual(a[0].tolist(), list(range(10)))

        a = tk.zeros((10, 10))
        index_x = tk.array([0, 2, 3, 7])
        index_y = tk.array([3, 3, 1, 2])
        u = tk.random.uniform(shape=(4,))
        a = a.at[index_x, index_y].add(u)
        self.assertTrue(tk.allclose(a.sum(), u.sum()))
        self.assertEqualArray(a.sum(), u.sum(), atol=1e-6, rtol=1e-5)
        self.assertEqual(a[index_x, index_y].tolist(), u.tolist())

        # Test all array.at ops
        a = tk.random.uniform(shape=(10, 5, 2))
        idx_x = tk.array([0, 4])
        update = tk.ones((2, 5))
        a[idx_x, :, 0] = 0
        a = a.at[idx_x, :, 0].add(update)
        self.assertEqualArray(a[idx_x, :, 0], update)
        a = a.at[idx_x, :, 0].subtract(update)
        self.assertEqualArray(a[idx_x, :, 0], tk.zeros_like(update))
        a = a.at[idx_x, :, 0].add(2 * update)
        self.assertEqualArray(a[idx_x, :, 0], 2 * update)
        a = a.at[idx_x, :, 0].multiply(2 * update)
        self.assertEqualArray(a[idx_x, :, 0], 4 * update)
        a = a.at[idx_x, :, 0].divide(3 * update)
        self.assertEqualArray(a[idx_x, :, 0], (4 / 3) * update)
        a[idx_x, :, 0] = 5
        update = tk.arange(10).reshape(2, 5)
        a = a.at[idx_x, :, 0].maximum(update)
        self.assertEqualArray(a[idx_x, :, 0], tk.maximum(a[idx_x, :, 0], update))
        a[idx_x, :, 0] = 5
        a = a.at[idx_x, :, 0].minimum(update)
        self.assertEqualArray(a[idx_x, :, 0], tk.minimum(a[idx_x, :, 0], update))

        update = tk.array([1.0, 2.0])[None, None, None]
        src = tk.array([1.0, 2.0])[None, :]
        src = src.at[0:1].add(update)
        self.assertTrue(tk.array_equal(src, tk.array([[2.0, 4.0]])))

        # Test all array.at ops with slice-only indices
        a = tk.random.uniform(shape=(10, 5, 2))
        update = tk.ones((2, 5))
        a[1:3, :, 0] = 0
        a = a.at[1:3, :, 0].add(update)
        self.assertEqualArray(a[1:3, :, 0], update)
        a = a.at[1:3, :, 0].subtract(update)
        self.assertEqualArray(a[1:3, :, 0], tk.zeros_like(update))
        a = a.at[1:3, :, 0].add(2 * update)
        self.assertEqualArray(a[1:3, :, 0], 2 * update)
        a = a.at[1:3, :, 0].multiply(2 * update)
        self.assertEqualArray(a[1:3, :, 0], 4 * update)
        a = a.at[1:3, :, 0].divide(3 * update)
        self.assertEqualArray(a[1:3, :, 0], (4 / 3) * update)
        a[1:3, :, 0] = 5
        update = tk.arange(10).reshape(2, 5)
        a = a.at[1:3, :, 0].maximum(update)
        self.assertEqualArray(a[1:3, :, 0], tk.maximum(a[1:3, :, 0], update))
        a[1:3, :, 0] = 5
        a = a.at[1:3, :, 0].minimum(update)
        self.assertEqualArray(a[1:3, :, 0], tk.minimum(a[1:3, :, 0], update))

    @unittest.skipIf(not tk.is_available(tk.gpu), "No GPU available")
    def test_array_at_complex_add_gpu(self):
        n = 4096
        base = [1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j]

        with tk.stream(tk.gpu):
            a = tk.array(base, dtype=tk.complex64)
            update_indices = tk.full((n,), 3, dtype=tk.int32)
            updates = tk.full((n,), 1 + 3j, dtype=tk.complex64)
            out = a.at[update_indices].add(updates)
            tk.eval(out)

            indices = tk.array([1, 1, 3])
            x = tk.array([1 + 0j, 3 + 4j, 6 + 8j, 5 + 12j], dtype=tk.complex64)

            def loss(z):
                return tk.square(tk.abs(z[indices])).sum()

            _, gradient = tk.value_and_grad(loss)(x)
            tk.eval(gradient)

        expected = base.copy()
        expected[-1] += n * (1 + 3j)
        self.assertEqual(out.tolist(), expected)
        np.testing.assert_allclose(
            np.array(gradient),
            np.array([0, 12 + 16j, 0, 10 + 24j], dtype=np.complex64),
            rtol=0,
            atol=1e-5,
        )

    def test_array_at_slice_update_extensive(self):
        # Test with transposed inputs
        a = tk.zeros((4, 5))
        update = tk.ones((5, 2)).T  # Shape (2, 5)
        a = a.at[1:3, :].add(update)
        self.assertEqualArray(a[1:3, :], update)

        # Test with transposed updates on transposed slice
        a = tk.zeros((5, 4))
        update = tk.ones((2, 5))
        a = a.at[:, 1:3].add(update.T)
        self.assertEqualArray(a[:, 1:3], update.T)

        # Test with slice of another array as update
        source = tk.arange(20, dtype=tk.float32).reshape(4, 5)
        a = tk.zeros((4, 5))
        update = source[1:3, :]  # Shape (2, 5)
        a = a.at[0:2, :].add(update)
        self.assertEqualArray(a[0:2, :], source[1:3, :])

        # Test with both input and update being slices
        source = tk.arange(30, dtype=tk.float32).reshape(5, 6)
        a = tk.zeros((5, 6))
        a = a.at[1:4, 1:5].add(source[0:3, 0:4])
        self.assertEqualArray(a[1:4, 1:5], source[0:3, 0:4])

        # Test with transposed slice of another array
        source = tk.arange(20, dtype=tk.float32).reshape(4, 5)
        a = tk.zeros((5, 4))
        update = source[1:3, :].T  # Shape (5, 2)
        a = a.at[:, 1:3].add(update)
        self.assertEqualArray(a[:, 1:3], update)

        # Test with negative indexing in slices
        a = tk.zeros((5, 5))
        update = tk.ones((2, 5))
        a = a.at[-3:-1, :].add(update)
        self.assertEqualArray(a[-3:-1, :], update)

        # Test with strided slices
        a = tk.zeros((6, 6))
        update = tk.ones((2, 3))
        a = a.at[1:5:2, 0:6:2].add(update)
        self.assertEqualArray(a[1:5:2, 0:6:2], update)

        # Test with slice of transposed array
        source = tk.arange(20, dtype=tk.float32).reshape(4, 5)
        a = tk.zeros((5, 4))
        update = source.T[:, 1:3]  # Shape (5, 2)
        a = a.at[:, 1:3].add(update)
        self.assertEqualArray(a[:, 1:3], update)

        # Test with 3D arrays and transposed updates
        a = tk.zeros((3, 4, 5))
        update = tk.ones((4, 3, 5)).transpose(1, 0, 2)  # Shape (3, 4, 5)
        a = a.at[:, :, :].add(update)
        self.assertEqualArray(a, update)

        # Test with slice of 3D array
        source = tk.arange(60, dtype=tk.float32).reshape(3, 4, 5)
        a = tk.zeros((3, 4, 5))
        update = source[0:2, :, :]
        a = a.at[1:3, :, :].add(update)
        self.assertEqualArray(a[1:3, :, :], source[0:2, :, :])

        # Test with mixed slice and index
        a = tk.zeros((4, 5, 6))
        update = tk.ones((2, 6))
        a = a.at[1:3, 2, :].add(update)
        self.assertEqualArray(a[1:3, 2, :], update)

        # Test with update from strided slice
        source = tk.arange(60, dtype=tk.float32).reshape(3, 4, 5)
        a = tk.zeros((3, 2, 5))
        update = source[:, ::2, :]  # Shape (3, 2, 5)
        a = a.at[:, :, :].add(update)
        self.assertEqualArray(a, update)

    def test_slice_update_contiguous_2d(self):
        for shape in [(32, 32), (64, 64), (17, 33), (128, 128)]:
            for upd_shape in [(16, 16), (8, 8), (4, 4)]:
                if upd_shape[0] > shape[0] or upd_shape[1] > shape[1]:
                    continue
                y = tk.zeros(shape)
                x = tk.random.normal(upd_shape)
                z = y.at[: upd_shape[0], : upd_shape[1]].add(x)
                diff = z[: upd_shape[0], : upd_shape[1]] - x
                self.assertTrue(tk.allclose(diff, tk.zeros_like(diff)))

        # Test non-zero offset slice update
        y = tk.zeros((32, 32))
        x = tk.random.normal((16, 16))
        z = y.at[16:, 16:].add(x)
        self.assertTrue(tk.allclose(z[16:, 16:] - x, tk.zeros_like(x)))

        # Test with size divisible by 4, 2, and odd
        for cols in [32, 18, 15]:
            y = tk.zeros((32, cols))
            x = tk.random.normal((16, cols))
            z = y.at[:16, :].add(x)
            self.assertTrue(tk.allclose(z[:16, :] - x, tk.zeros_like(x)))

    def test_slice_negative_step(self):
        a_np = np.arange(20)
        a_mx = tk.array(a_np)

        # Basic negative slice
        b_np = a_np[::-1]
        b_mx = a_mx[::-1]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Bounds negative slice
        b_np = a_np[-3:3:-1]
        b_mx = a_mx[-3:3:-1]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Bounds negative slice
        b_np = a_np[25:-50:-1]
        b_mx = a_mx[25:-50:-1]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Jumping negative slice
        b_np = a_np[::-3]
        b_mx = a_mx[::-3]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Bounds and negative slice
        b_np = a_np[-3:3:-3]
        b_mx = a_mx[-3:3:-3]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Bounds and negative slice
        b_np = a_np[25:-50:-3]
        b_mx = a_mx[25:-50:-3]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Negative slice and ascending bounds
        b_np = a_np[0:20:-3]
        b_mx = a_mx[0:20:-3]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Multi-dim negative slices
        a_np = np.arange(3 * 6 * 4).reshape(3, 6, 4)
        a_mx = tk.array(a_np)

        # Flip each dim
        b_np = a_np[..., ::-1]
        b_mx = a_mx[..., ::-1]
        self.assertTrue(np.array_equal(b_np, b_mx))

        b_np = a_np[:, ::-1, :]
        b_mx = a_mx[:, ::-1, :]
        self.assertTrue(np.array_equal(b_np, b_mx))

        b_np = a_np[::-1, ...]
        b_mx = a_mx[::-1, ...]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Flip pairs of dims
        b_np = a_np[::-1, 1:5:2, ::-2]
        b_mx = a_mx[::-1, 1:5:2, ::-2]
        self.assertTrue(np.array_equal(b_np, b_mx))

        b_np = a_np[::-1, ::-2, 1:5:2]
        b_mx = a_mx[::-1, ::-2, 1:5:2]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Flip all dims
        b_np = a_np[::-1, ::-3, ::-2]
        b_mx = a_mx[::-1, ::-3, ::-2]
        self.assertTrue(np.array_equal(b_np, b_mx))

    def test_api(self):
        x = tk.array(np.random.rand(10, 10, 10))
        ops = [
            ("reshape", (100, -1)),
            "square",
            "sqrt",
            "rsqrt",
            "reciprocal",
            "exp",
            "log",
            "sin",
            "cos",
            "log1p",
            "abs",
            "log10",
            "log2",
            "conj",
            ("all", 1),
            ("any", 1),
            ("transpose", (0, 2, 1)),
            ("sum", 1),
            ("prod", 1),
            ("min", 1),
            ("max", 1),
            ("logcumsumexp", 1),
            ("logsumexp", 1),
            ("mean", 1),
            ("var", 1),
            ("argmin", 1),
            ("argmax", 1),
            ("cummax", 1),
            ("cummin", 1),
            ("cumprod", 1),
            ("cumsum", 1),
            ("diagonal", 0, 0, 1),
            ("flatten", 0, -1),
            ("moveaxis", 1, 2),
            ("round", 2),
            ("std", 1, True, 0),
            ("swapaxes", 1, 2),
        ]
        for op in ops:
            if isinstance(op, tuple):
                op, *args = op
            else:
                args = tuple()
            y1 = getattr(tk, op)(x, *args)
            y2 = getattr(x, op)(*args)
            self.assertEqual(y1.dtype, y2.dtype)
            self.assertEqual(y1.shape, y2.shape)
            self.assertTrue(tk.array_equal(y1, y2))

        y1 = tk.split(x, 2)
        y2 = x.split(2)
        self.assertEqual(len(y1), 2)
        self.assertEqual(len(y1), len(y2))
        self.assertTrue(tk.array_equal(y1[0], y2[0]))
        self.assertTrue(tk.array_equal(y1[1], y2[1]))
        x = tk.array(np.random.rand(10, 10, 1))
        y1 = tk.squeeze(x, axis=2)
        y2 = x.squeeze(axis=2)
        self.assertEqual(y1.shape, y2.shape)
        self.assertTrue(tk.array_equal(y1, y2))

    def test_memoryless_copy(self):
        a_mx = tk.ones((2, 2))
        b_mx = tk.broadcast_to(a_mx, (5, 2, 2))

        # Make np arrays without copy
        a_np = np.array(a_mx, copy=False)
        b_np = np.array(b_mx, copy=False)

        # Check that we get read-only array that does not own the underlying data
        self.assertFalse(a_np.flags.owndata)
        self.assertTrue(a_np.flags.writeable)

        # Check contents
        self.assertTrue(np.array_equal(np.ones((2, 2), dtype=np.float32), a_np))
        self.assertTrue(np.array_equal(np.ones((5, 2, 2), dtype=np.float32), b_np))

        # Check strides
        self.assertSequenceEqual(b_np.strides, (0, 8, 4))

    def test_np_array_conversion_copies_by_default(self):
        a_mx = tk.ones((2, 2))
        a_np = np.array(a_mx)
        self.assertTrue(a_np.flags.owndata)
        self.assertTrue(a_np.flags.writeable)

    def test_buffer_protocol(self):
        dtypes_list = [
            (tk.bool_, np.bool_, None),
            (tk.uint8, np.uint8, np.iinfo),
            (tk.uint16, np.uint16, np.iinfo),
            (tk.uint32, np.uint32, np.iinfo),
            (tk.uint64, np.uint64, np.iinfo),
            (tk.int8, np.int8, np.iinfo),
            (tk.int16, np.int16, np.iinfo),
            (tk.int32, np.int32, np.iinfo),
            (tk.int64, np.int64, np.iinfo),
            (tk.float16, np.float16, np.finfo),
            (tk.float32, np.float32, np.finfo),
            (tk.complex64, np.complex64, np.finfo),
        ]

        for tiki_dtype, np_dtype, info_fn in dtypes_list:
            a_np = np.random.uniform(low=0, high=100, size=(3, 4)).astype(np_dtype)
            if info_fn is not None:
                info = info_fn(np_dtype)
                a_np[0, 0] = info.min
                a_np[0, 1] = info.max
            a_mx = tk.array(a_np)
            for f in [lambda x: x, lambda x: x.T]:
                mv_mx = memoryview(f(a_mx))
                mv_np = memoryview(f(a_np))
                self.assertEqual(mv_mx.strides, mv_np.strides, f"{tiki_dtype}{np_dtype}")
                self.assertEqual(mv_mx.shape, mv_np.shape, f"{tiki_dtype}{np_dtype}")
                # correct buffer format for 8 byte (unsigned) 'long long' is Q/q, see
                # https://docs.python.org/3.10/library/struct.html#format-characters
                # numpy returns L/l, as 'long' is equivalent to 'long long' on 64bit machines, so q and l are equivalent
                # see https://github.com/pybind/pybind11/issues/1908
                if np_dtype == np.uint64:
                    self.assertEqual(mv_mx.format, "Q", f"{tiki_dtype}{np_dtype}")
                elif np_dtype == np.int64:
                    self.assertEqual(mv_mx.format, "q", f"{tiki_dtype}{np_dtype}")
                # for windows long is 32bit and numpy returns L/l.
                elif np_dtype == np.uint32 and platform.system() == "Windows":
                    self.assertEqual(mv_mx.format, "I", f"{tiki_dtype}{np_dtype}")
                elif np_dtype == np.int32 and platform.system() == "Windows":
                    self.assertEqual(mv_mx.format, "i", f"{tiki_dtype}{np_dtype}")
                else:
                    self.assertEqual(
                        mv_mx.format, mv_np.format, f"{tiki_dtype}{np_dtype}"
                    )
                self.assertFalse(mv_mx.readonly)
                back_to_npy = np.array(mv_mx, copy=False)
                self.assertEqualArray(
                    back_to_npy,
                    f(a_np),
                    atol=0,
                    rtol=0,
                )

        # extra test for bfloat16, which is not numpy convertible
        a_mx = tk.random.uniform(low=0, high=100, shape=(3, 4), dtype=tk.bfloat16)
        mv_mx = memoryview(a_mx)
        self.assertEqual(mv_mx.strides, (8, 2))
        self.assertEqual(mv_mx.shape, (3, 4))
        self.assertIn(mv_mx.format, "bfloat16")
        with self.assertRaises(ValueError) as cm:
            np.array(a_mx)
        self.assertIn("bfloat16", str(cm.exception))

        # Test buffer protocol with non-arrays ie bytes
        a = ord("a") * 257 + tk.arange(10).astype(tk.int16)
        ab = bytes(a)
        self.assertEqual(len(ab), 20)
        if sys.byteorder == "little":
            self.assertEqual(b"aaaaaaaaaa", ab[1::2])
            self.assertEqual(b"abcdefghij", ab[::2])
        else:
            self.assertEqual(b"aaaaaaaaaa", ab[::2])
            self.assertEqual(b"abcdefghij", ab[1::2])

    def test_buffer_protocol_ref_counting(self):
        a = tk.arange(3)
        wr = weakref.ref(a)
        self.assertIsNotNone(wr())
        mv = memoryview(a)
        a = None
        self.assertIsNotNone(wr())
        mv = None
        self.assertIsNone(wr())

    def test_array_view_ref_counting(self):
        a = tk.arange(3)
        wr = weakref.ref(a)
        self.assertIsNotNone(wr())
        a_np = np.array(a, copy=False)
        a = None
        self.assertIsNotNone(wr())
        a_np = None
        self.assertIsNone(wr())

    def test_create_from_buffer(self):
        x = tk.array(b"Hello")
        self.assertEqual(x.dtype, tk.uint8)
        self.assertEqual(x.tolist(), [72, 101, 108, 108, 111])

        x = tk.array(bytearray([1, 2, 3]))
        self.assertEqual(x.dtype, tk.uint8)
        self.assertEqual(x.tolist(), [1, 2, 3])

    @unittest.skipIf(not has_tf, "requires TensorFlow")
    def test_buffer_protocol_tf(self):
        dtypes_list = [
            (
                tk.bool_,
                tf.bool,
                np.bool_,
            ),
            (
                tk.uint8,
                tf.uint8,
                np.uint8,
            ),
            (
                tk.uint16,
                tf.uint16,
                np.uint16,
            ),
            (
                tk.uint32,
                tf.uint32,
                np.uint32,
            ),
            (tk.uint64, tf.uint64, np.uint64),
            (tk.int8, tf.int8, np.int8),
            (tk.int16, tf.int16, np.int16),
            (tk.int32, tf.int32, np.int32),
            (tk.int64, tf.int64, np.int64),
            (tk.float16, tf.float16, np.float16),
            (tk.float32, tf.float32, np.float32),
            (
                tk.complex64,
                tf.complex64,
                np.complex64,
            ),
        ]

        for tiki_dtype, tf_dtype, np_dtype in dtypes_list:
            a_np = np.random.uniform(low=0, high=100, size=(3, 4)).astype(np_dtype)
            a_tf = tf.constant(a_np, dtype=tf_dtype)
            a_mx = tk.array(np.array(a_tf))
            for f in [
                lambda x: x,
                lambda x: tf.transpose(x) if isinstance(x, tf.Tensor) else x.T,
            ]:
                mv_mx = memoryview(f(a_mx))
                mv_tf = memoryview(f(a_tf))
                if (mv_mx.c_contiguous and mv_tf.c_contiguous) or (
                    mv_mx.f_contiguous and mv_tf.f_contiguous
                ):
                    self.assertEqual(
                        mv_mx.strides, mv_tf.strides, f"{tiki_dtype}{tf_dtype}"
                    )
                self.assertEqual(mv_mx.shape, mv_tf.shape, f"{tiki_dtype}{tf_dtype}")
                self.assertFalse(mv_mx.readonly)
                back_to_npy = np.array(mv_mx)
                self.assertEqualArray(
                    back_to_npy,
                    f(a_tf),
                    atol=0,
                    rtol=0,
                )

    def test_logical_overloads(self):
        with self.assertRaises(ValueError):
            tk.array(1.0) & tk.array(1)
        with self.assertRaises(ValueError):
            tk.array(1.0) | tk.array(1)

        self.assertEqual((tk.array(True) & True).item(), True)
        self.assertEqual((tk.array(True) & False).item(), False)
        self.assertEqual((tk.array(True) | False).item(), True)
        self.assertEqual((tk.array(False) | False).item(), False)
        self.assertEqual((~tk.array(False)).item(), True)
        self.assertEqual((tk.array(False) ^ True).item(), True)

    def test_inplace(self):
        iops = [
            "__iadd__",
            "__isub__",
            "__imul__",
            "__ifloordiv__",
            "__imod__",
            "__ipow__",
            "__ixor__",
        ]

        for op in iops:
            a = tk.array([1, 2, 3])
            a_np = np.array(a)
            b = a
            b = getattr(a, op)(3)
            self.assertTrue(tk.array_equal(a, b))
            out_np = getattr(a_np, op)(3)
            self.assertTrue(np.array_equal(out_np, a))

        with self.assertRaises(ValueError):
            a = tk.array([1])
            a /= 1

        a = tk.array([2.0])
        b = a
        b /= 2
        self.assertEqual(b.item(), 1.0)
        self.assertEqual(b.item(), a.item())

        a = tk.array(True)
        b = a
        b &= False
        self.assertEqual(b.item(), False)
        self.assertEqual(b.item(), a.item())

        a = tk.array(False)
        b = a
        b |= True
        self.assertEqual(b.item(), True)
        self.assertEqual(b.item(), a.item())

        # In-place matmul on its own
        a = tk.array([[1.0, 2.0], [3.0, 4.0]])
        b = a
        b @= a
        self.assertTrue(tk.array_equal(a, b))

        a = tk.array(False)
        a ^= True
        self.assertEqual(a.item(), True)

    def test_inplace_preserves_ids(self):
        a = tk.array([1.0])
        orig_id = id(a)
        a += tk.array(2.0)
        self.assertEqual(id(a), orig_id)

        a[0] = 2.0
        self.assertEqual(id(a), orig_id)

        a -= tk.array(3.0)
        self.assertEqual(id(a), orig_id)

        a *= tk.array(3.0)
        self.assertEqual(id(a), orig_id)

    def test_load_from_pickled_np(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = pickle.loads(pickle.dumps(a))
        self.assertTrue(tk.array_equal(tk.array(a), tk.array(b)))

        a = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        b = pickle.loads(pickle.dumps(a))
        self.assertTrue(tk.array_equal(tk.array(a), tk.array(b)))

    def test_multi_output_leak(self):
        def fun():
            a = tk.zeros((2**20))
            tk.eval(a)
            b, c = tk.divmod(a, a)
            del b, c

        fun()
        tk.synchronize()
        peak_1 = tk.get_peak_memory()
        fun()
        tk.synchronize()
        peak_2 = tk.get_peak_memory()
        self.assertEqual(peak_1, peak_2)

        def fun():
            a = tk.array([1.0, 2.0, 3.0, 4.0])
            b, _ = tk.divmod(a, a)
            return tk.log(b)

        fun()
        tk.synchronize()
        peak_1 = tk.get_peak_memory()
        fun()
        tk.synchronize()
        peak_2 = tk.get_peak_memory()
        self.assertEqual(peak_1, peak_2)

    def test_add_numpy(self):
        x = tk.array(1)
        y = np.array(2, dtype=np.int32)
        z = x + y
        self.assertEqual(z.dtype, tk.int32)
        self.assertEqual(z.item(), 3)

    def test_dlpack(self):
        class CpuDLPack:
            def __init__(self, array):
                self.array = array

            def __dlpack_device__(self):
                return (1, 0)

            def __dlpack__(self, *args, **kwargs):
                kwargs["dl_device"] = (1, 0)
                return self.array.__dlpack__(*args, **kwargs)

        x = tk.array(1, dtype=tk.int32)
        y = np.from_dlpack(CpuDLPack(x))
        self.assertTrue(tk.array_equal(y, x))

        x = tk.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.from_dlpack(CpuDLPack(x))
        self.assertTrue(tk.array_equal(y, x))

        x = tk.arange(16).reshape(4, 4)
        x = x[::2, ::2]
        y = np.from_dlpack(CpuDLPack(x))
        self.assertTrue(tk.array_equal(y, x))

    def test_from_dlpack_cpu(self):
        x = np.arange(3, dtype=np.float32)

        # copy=None may adopt the buffer or copy; either way the values match
        # the source at import time.
        y = tk.from_dlpack(x)
        self.assertEqual(y.tolist(), [0.0, 1.0, 2.0])

        # copy=True always copies, so later mutations of the source are not seen.
        y = tk.from_dlpack(x, copy=True)
        x += 10
        self.assertEqual(y.tolist(), [0.0, 1.0, 2.0])

        # copy=False adopts the buffer when possible and raises otherwise; it
        # must never silently copy.
        x = np.arange(3, dtype=np.float32)
        try:
            y = tk.from_dlpack(x, copy=False)
            x += 10
        except ValueError:
            pass
        else:
            self.assertEqual(y.tolist(), [10.0, 11.0, 12.0])

    def test_dlpack_cpu_dtype_mapping(self):
        class CpuDLPack:
            def __init__(self, array):
                self.array = array

            def __dlpack_device__(self):
                return (1, 0)

            def __dlpack__(self, *args, **kwargs):
                kwargs["dl_device"] = (1, 0)
                return self.array.__dlpack__(*args, **kwargs)

        dlpack_to_tiki = [
            (np.bool_, tk.bool_),
            (np.uint8, tk.uint8),
            (np.uint16, tk.uint16),
            (np.uint32, tk.uint32),
            (np.uint64, tk.uint64),
            (np.int8, tk.int8),
            (np.int16, tk.int16),
            (np.int32, tk.int32),
            (np.int64, tk.int64),
            (np.float16, tk.float16),
            (np.float32, tk.float32),
            (np.float64, tk.float32),
            (np.complex64, tk.complex64),
            (np.complex128, tk.complex64),
        ]
        for np_dtype, tiki_dtype in dlpack_to_tiki:
            with self.subTest(direction="import", dtype=np_dtype):
                x = np.ones(3, dtype=np_dtype)
                y = tk.from_dlpack(x)
                self.assertEqual(y.dtype, tiki_dtype)

        if torch is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                x = torch.ones(3, dtype=torch.complex32)
            with self.assertRaises(ValueError):
                tk.from_dlpack(x)

        tiki_to_dlpack = [
            (tk.bool_, np.bool_),
            (tk.uint8, np.uint8),
            (tk.uint16, np.uint16),
            (tk.uint32, np.uint32),
            (tk.uint64, np.uint64),
            (tk.int8, np.int8),
            (tk.int16, np.int16),
            (tk.int32, np.int32),
            (tk.int64, np.int64),
            (tk.float16, np.float16),
            (tk.float32, np.float32),
            (tk.complex64, np.complex64),
        ]
        for tiki_dtype, np_dtype in tiki_to_dlpack:
            with self.subTest(direction="export", dtype=tiki_dtype):
                x = tk.ones((3,), dtype=tiki_dtype)
                y = np.from_dlpack(CpuDLPack(x))
                self.assertEqual(y.dtype, np_dtype)

        if torch is not None and has_torch_mps:
            x = tk.ones((3,), dtype=tk.bfloat16)
            y = torch.from_dlpack(x)
            self.assertEqual(y.dtype, torch.bfloat16)

    def test_from_dlpack_cpu_strided(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        view = x.T
        y = tk.from_dlpack(view)

        self.assertEqual(y.tolist(), view.tolist())
        self.assertFalse(memoryview(y).c_contiguous)
        self.assertEqual(memoryview(y).strides, view.strides)

        stepped = np.arange(20, dtype=np.int32)[2:10:2]
        y = tk.from_dlpack(stepped)
        self.assertEqual(y.tolist(), [2, 4, 6, 8])
        self.assertFalse(memoryview(y).c_contiguous)
        self.assertEqual(memoryview(y).strides, stepped.strides)

        broadcast = np.broadcast_to(np.array([7], dtype=np.int32), (3,))
        y = tk.from_dlpack(broadcast)
        self.assertEqual(y.tolist(), [7, 7, 7])
        self.assertFalse(memoryview(y).c_contiguous)
        self.assertEqual(memoryview(y).strides, broadcast.strides)

        negative_stride = np.arange(5, dtype=np.float32)[::-1]
        with self.assertRaises(ValueError):
            tk.from_dlpack(negative_stride)

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_dlpack_import(self):
        assert torch is not None
        x = torch.arange(12, device="mps", dtype=torch.float32).reshape(3, 4)
        self.assertEqual(x.__dlpack_device__()[0], 8)

        y = tk.asarray(x)
        self.assertEqual(y.dtype, tk.float32)
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), x.cpu().numpy().tolist())
        self.assertIn("array(", repr(y))
        mv = memoryview(y)
        self.assertEqual(mv.tolist(), x.cpu().numpy().tolist())

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_array_copies_dlpack_input(self):
        assert torch is not None
        x = torch.arange(3, device="mps", dtype=torch.float32)
        torch.mps.synchronize()
        y = tk.array(x)

        x.zero_()
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), [0.0, 1.0, 2.0])

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_asarray_copy_true_copies_dlpack_input(self):
        assert torch is not None
        x = torch.arange(3, device="mps", dtype=torch.float32)
        torch.mps.synchronize()
        y = tk.asarray(x, copy=True)

        x.zero_()
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), [0.0, 1.0, 2.0])

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_dlpack_zero_copy_shares_updates(self):
        assert torch is not None
        x = torch.arange(12, device="mps", dtype=torch.float32).reshape(3, 4)
        torch.mps.synchronize()
        y = tk.asarray(x)

        x.zero_()
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), x.cpu().numpy().tolist())

        y += 10
        tk.eval(y)
        self.assertEqual(x.cpu().numpy().tolist(), y.tolist())

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_dlpack_matching_dtype_argument_shares_updates(self):
        assert torch is not None
        x = torch.arange(12, device="mps", dtype=torch.float32).reshape(3, 4)
        torch.mps.synchronize()
        y = tk.asarray(x, dtype=tk.float32, copy=False)
        self.assertEqual(y.dtype, tk.float32)

        x.zero_()
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), x.cpu().numpy().tolist())

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_dlpack_different_dtype_argument_copies(self):
        assert torch is not None
        x = torch.arange(12, device="mps", dtype=torch.float32).reshape(3, 4)
        torch.mps.synchronize()
        z = tk.asarray(x, dtype=tk.float16)
        expected = x.to(torch.float16).cpu().numpy().tolist()

        self.assertEqual(z.dtype, tk.float16)
        self.assertEqual(z.tolist(), expected)

        x.zero_()
        torch.mps.synchronize()
        self.assertEqual(z.tolist(), expected)

        with self.assertRaises(ValueError):
            tk.asarray(x, dtype=tk.float16, copy=False)

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_dlpack_data_offset(self):
        assert torch is not None
        view = torch.arange(12, device="mps", dtype=torch.float32)[3:9]
        view_mx = tk.asarray(view)
        torch.mps.synchronize()
        self.assertEqual(view_mx.tolist(), view.cpu().numpy().tolist())

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_dlpack_strided_view(self):
        assert torch is not None
        x = torch.arange(12, device="mps", dtype=torch.float32).reshape(3, 4)
        view = x.T
        torch.mps.synchronize()
        y = tk.asarray(view, copy=False)
        self.assertEqual(y.tolist(), view.cpu().numpy().tolist())

        x[0, 1] = 99
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), view.cpu().numpy().tolist())

        y_copy = tk.asarray(view, copy=True)
        expected = view.cpu().numpy().tolist()
        self.assertFalse(memoryview(y_copy).c_contiguous)
        self.assertEqual(
            memoryview(y_copy).strides,
            tuple(s * view.element_size() for s in view.stride()),
        )
        x[0, 2] = 77
        torch.mps.synchronize()
        self.assertEqual(y_copy.tolist(), expected)

        z = tk.asarray(view, dtype=tk.float16)
        self.assertEqual(z.dtype, tk.float16)
        self.assertFalse(memoryview(z).c_contiguous)
        self.assertEqual(memoryview(z).strides, tuple(s * 2 for s in view.stride()))

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_dlpack_stepped_view(self):
        x = torch.arange(20, device="mps", dtype=torch.int32)
        view = x[2:10:2]
        torch.mps.synchronize()
        y = tk.asarray(view, copy=False)
        self.assertEqual(y.tolist(), [2, 4, 6, 8])

        x[4] = 99
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), [2, 99, 6, 8])

        y_copy = tk.asarray(view, copy=True)
        expected = y.tolist()
        x[6] = 77
        torch.mps.synchronize()
        self.assertEqual(y_copy.tolist(), expected)

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_dlpack_broadcast_stride(self):
        assert torch is not None
        x = torch.tensor([7], device="mps", dtype=torch.int32)
        view = x.expand(3)
        torch.mps.synchronize()
        y = tk.asarray(view, copy=False)
        self.assertEqual(y.tolist(), [7, 7, 7])

        x.zero_()
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), [0, 0, 0])

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_dlpack_bfloat16(self):
        assert torch is not None
        x = torch.arange(12, device="mps", dtype=torch.float32).reshape(3, 4)
        bf = x.to(torch.bfloat16)
        bf_mx = tk.asarray(bf)

        self.assertEqual(bf_mx.dtype, tk.bfloat16)
        torch.mps.synchronize()
        self.assertEqual(
            bf_mx.astype(tk.float32).tolist(),
            bf.to(torch.float32).cpu().numpy().tolist(),
        )

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_torch_mps_array_operand(self):
        assert torch is not None
        a = tk.array([1])
        b = torch.tensor([2])
        self.assertTrue(tk.array_equal(a + b, tk.array([3])))

        b_mps = b.to("mps")
        torch.mps.synchronize()
        self.assertTrue(tk.array_equal(a + b_mps, tk.array([3])))

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_tiki_dlpack_exports_mps_tensor_to_torch(self):
        assert torch is not None
        x = tk.array([1]).astype(tk.float16)
        tk.eval(x)
        y = torch.utils.dlpack.from_dlpack(x)
        torch.mps.synchronize()

        self.assertEqual(y.device.type, "mps")
        self.assertEqual(y.dtype, torch.float16)
        self.assertEqual(y.cpu().numpy().tolist(), [1.0])

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_tiki_dlpack_exports_mps_tensor_to_torch_tensor(self):
        assert torch is not None
        x = tk.array([1]).astype(tk.float16)
        tk.eval(x)
        y = torch.tensor(x)
        torch.mps.synchronize()

        self.assertEqual(y.device.type, "mps")
        self.assertEqual(y.dtype, torch.float16)
        self.assertEqual(y.cpu().numpy().tolist(), [1.0])

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_tiki_dlpack_export_torch_update_writes_tiki_buffer(self):
        x = tk.arange(8, dtype=tk.float32)
        y = x[2:6]
        tk.eval(y)
        t = torch.utils.dlpack.from_dlpack(y)

        self.assertEqual(t.device.type, "mps")
        self.assertEqual(t.cpu().numpy().tolist(), [2.0, 3.0, 4.0, 5.0])

        t.zero_()
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(x.tolist(), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0])

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_from_dlpack_torch_mps_copy_none_shares_updates(self):
        assert torch is not None
        x = torch.arange(3, device="mps", dtype=torch.float32)
        torch.mps.synchronize()
        y = tk.from_dlpack(x)

        x.zero_()
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), [0.0, 0.0, 0.0])

        y += 10
        tk.eval(y)
        self.assertEqual(x.cpu().numpy().tolist(), [10.0, 10.0, 10.0])

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_from_dlpack_torch_mps_copy_false_shares_updates(self):
        assert torch is not None
        x = torch.arange(3, device="mps", dtype=torch.float32)
        torch.mps.synchronize()
        y = tk.from_dlpack(x, copy=False)

        x.zero_()
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), [0.0, 0.0, 0.0])

    @unittest.skipUnless(has_torch_mps, "PyTorch MPS is required")
    def test_from_dlpack_torch_mps_copy_true_copies(self):
        assert torch is not None
        x = torch.arange(3, device="mps", dtype=torch.float32)
        torch.mps.synchronize()
        y = tk.from_dlpack(x, copy=True)

        x.zero_()
        torch.mps.synchronize()
        self.assertEqual(y.tolist(), [0.0, 1.0, 2.0])

    def test_getitem_with_list(self):
        a = tk.array([1, 2, 3, 4, 5])
        idx = [0, 2, 4]
        self.assertTrue(np.array_equal(a[idx], np.array(a)[idx]))

        a = tk.array([[1, 2], [3, 4], [5, 6]])
        idx = [0, 2]
        self.assertTrue(np.array_equal(a[idx], np.array(a)[idx]))

        a = tk.arange(10).reshape(5, 2)
        idx = [0, 2, 4]
        self.assertTrue(np.array_equal(a[idx], np.array(a)[idx]))

        idx = [0, 2]
        a = tk.arange(16).reshape(4, 4)
        anp = np.array(a)
        self.assertTrue(np.array_equal(a[idx, 0], anp[idx, 0]))
        self.assertTrue(np.array_equal(a[idx, :], anp[idx, :]))
        self.assertTrue(np.array_equal(a[0, idx], anp[0, idx]))
        self.assertTrue(np.array_equal(a[:, idx], anp[:, idx]))

    def test_setitem_with_list(self):
        a = tk.array([1, 2, 3, 4, 5])
        anp = np.array(a)
        idx = [0, 2, 4]
        a[idx] = 3
        anp[idx] = 3
        self.assertTrue(np.array_equal(a, anp))

        a = tk.array([[1, 2], [3, 4], [5, 6]])
        idx = [0, 2]
        anp = np.array(a)
        a[idx] = 3
        anp[idx] = 3
        self.assertTrue(np.array_equal(a, anp))

        a = tk.arange(10).reshape(5, 2)
        idx = [0, 2, 4]
        anp = np.array(a)
        a[idx] = 3
        anp[idx] = 3
        self.assertTrue(np.array_equal(a, anp))

        idx = [0, 2]
        a = tk.arange(16).reshape(4, 4)
        anp = np.array(a)
        a[idx, 0] = 1
        anp[idx, 0] = 1
        self.assertTrue(np.array_equal(a, anp))

        a[idx, :] = 2
        anp[idx, :] = 2
        self.assertTrue(np.array_equal(a, anp))

        a[0, idx] = 3
        anp[0, idx] = 3
        self.assertTrue(np.array_equal(a, anp))

        a[:, idx] = 4
        anp[:, idx] = 4
        self.assertTrue(np.array_equal(a, anp))

    def test_setitem_with_boolean_mask(self):
        # Python list mask
        a = tk.array([1.0, 2.0, 3.0])
        mask = [True, False, True]
        src = tk.array([5.0, 6.0])
        expected = tk.array([5.0, 2.0, 6.0])
        a[mask] = src
        self.assertTrue(tk.array_equal(a, expected))

        # tk.array scalar mask
        a = tk.array([1.0, 2.0, 3.0])
        mask = tk.array(True)
        expected = tk.array([5.0, 5.0, 5.0])
        a[mask] = 5.0
        self.assertTrue(tk.array_equal(a, expected))

        # scalar mask
        a = tk.array([1.0, 2.0, 3.0])
        mask = True
        expected = tk.array([5.0, 5.0, 5.0])
        a[mask] = 5.0
        self.assertTrue(tk.array_equal(a, expected))

        mask_np = np.zeros((1, 10, 10), dtype=bool)
        with self.assertRaises(ValueError):
            tk.arange(1000).reshape(10, 10, 10)[mask_np] = 0

        mask_np = np.zeros((10, 10, 1), dtype=bool)
        with self.assertRaises(ValueError):
            tk.arange(1000).reshape(10, 10, 10)[mask_np] = 0

    def test_array_namespace(self):
        a = tk.array(1.0)
        api = a.__array_namespace__()
        self.assertTrue(hasattr(api, "array"))
        self.assertTrue(hasattr(api, "add"))

    def test_array_namespace_asarray(self):
        xp = tk.array(1.0).__array_namespace__()
        self.assertTrue(hasattr(xp, "asarray"))

        arr = xp.asarray([1, 2, 3])
        self.assertEqual(arr.tolist(), [1, 2, 3])

        arr_f32 = xp.asarray([1, 2, 3], dtype=tk.float32)
        self.assertEqual(arr_f32.dtype, tk.float32)

        existing = tk.array([4, 5, 6])
        arr_pass = xp.asarray(existing)
        self.assertEqual(arr_pass.tolist(), [4, 5, 6])

    def test_asarray_copy(self):
        existing = tk.array([1, 2, 3])

        self.assertEqual(tk.asarray(existing, copy=True).tolist(), [1, 2, 3])
        self.assertEqual(
            tk.asarray(existing, dtype=tk.float32, copy=True).dtype, tk.float32
        )
        with self.assertRaises(ValueError):
            tk.asarray(existing, copy=False)
        with self.assertRaises(ValueError):
            tk.asarray(existing, dtype=tk.float32, copy=False)

    def test_asarray(self):
        # List inputs
        self.assertEqual(tk.asarray([1, 2, 3]).tolist(), [1, 2, 3])
        self.assertEqual(tk.asarray([[1, 2], [3, 4]]).tolist(), [[1, 2], [3, 4]])

        # Tuple inputs
        self.assertEqual(tk.asarray((1, 2, 3)).tolist(), [1, 2, 3])
        self.assertEqual(tk.asarray(((1, 2), (3, 4))).tolist(), [[1, 2], [3, 4]])

        # Mixed nesting
        self.assertEqual(tk.asarray([(1, 2), (3, 4)]).tolist(), [[1, 2], [3, 4]])
        self.assertEqual(tk.asarray(([1, 2], [3, 4])).tolist(), [[1, 2], [3, 4]])

        # Scalar inputs
        self.assertEqual(tk.asarray(42).item(), 42)
        self.assertEqual(tk.asarray(3.14).item(), 3.140000104904175)
        self.assertEqual(tk.asarray(True).item(), True)
        self.assertEqual(tk.asarray(1 + 2j).item(), (1 + 2j))

        # Tiki array inputs
        arr = tk.array([1, 2, 3])
        self.assertEqual(tk.asarray(arr).tolist(), [1, 2, 3])
        self.assertEqual(tk.asarray(arr, copy=True).tolist(), [1, 2, 3])
        with self.assertRaises(ValueError):
            tk.asarray(arr, copy=False)

        arr_int = tk.array([1, 2, 3], dtype=tk.int32)
        arr_float = tk.asarray(arr_int, dtype=tk.float32)
        self.assertEqual(arr_float.dtype, tk.float32)
        self.assertEqual(arr_float.tolist(), [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            tk.asarray(arr_int, dtype=tk.float32, copy=False)

        # NumPy array inputs
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mx_arr = tk.asarray(np_arr)
        self.assertEqual(mx_arr.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(mx_arr.dtype, tk.float32)
        # copy=False adopts the buffer when possible and raises otherwise; it
        # must never silently copy.
        try:
            mx_arr = tk.asarray(np_arr, copy=False)
        except ValueError:
            pass
        else:
            self.assertEqual(mx_arr.tolist(), [1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            tk.asarray([1, 2, 3], copy=False)

        # dtype parameter
        self.assertEqual(tk.asarray([1, 2, 3], dtype=tk.float32).dtype, tk.float32)
        self.assertEqual(tk.asarray(42, dtype=tk.float16).dtype, tk.float16)

    def test_to_scalar(self):
        a = tk.array(1)
        self.assertEqual(int(a), 1)
        self.assertEqual(float(a), 1)
        self.assertEqual(complex(a), 1 + 0j)

        a = tk.array(1.5)
        self.assertEqual(float(a), 1.5)
        self.assertEqual(int(a), 1)
        self.assertEqual(complex(a), 1.5 + 0j)

        a = tk.array(1 + 2j, dtype=tk.complex64)  # type: ignore
        self.assertEqual(complex(a), 1 + 2j)

        a = tk.zeros((2, 1))
        with self.assertRaises(ValueError):
            float(a)
        with self.assertRaises(ValueError):
            int(a)
        with self.assertRaises(ValueError):
            complex(a)

    def test_format(self):
        a = tk.arange(3)
        self.assertEqual(f"{a[0]:.2f}", "0.00")

        b = tk.array(0.35487)
        self.assertEqual(f"{b:.1f}", "0.4")

        with self.assertRaises(TypeError):
            s = f"{a:.2f}"

        a = tk.array([1, 2, 3])
        self.assertEqual(f"{a}", "array([1, 2, 3], dtype=int32)")

    def test_deep_graphs(self):
        # The following tests should simply run cleanly without a segfault or
        # crash due to exceeding recursion depth limits.

        # Deep graph destroyed without eval
        x = tk.array([1.0, 2.0])
        for _ in range(100_000):
            x = tk.sin(x)
        del x

        # Duplicate input deep graph destroyed without eval
        x = tk.array([1.0, 2.0])
        for _ in range(100_000):
            x = x + x

        # Deep graph with siblings destroyed without eval
        x = tk.array([1, 2])
        for _ in range(100_000):
            x = tk.concatenate(tk.split(x, 2))
        del x

        # Deep graph with eval
        x = tk.array([1.0, 2.0])
        for _ in range(100_000):
            x = tk.sin(x)
        tk.eval(x)

    def test_scalar_integer_conversion_overflow(self):
        y = tk.array(2000000000, dtype=tk.int32)
        x = 3000000000
        with self.assertRaises(ValueError):
            y + x
        with self.assertRaises(ValueError):
            tk.add(y, x)

    def test_real_imag(self):
        x = tk.array([1.0])
        self.assertEqual(x.real.item(), 1.0)
        self.assertEqual(x.imag.item(), 0.0)

        x = tk.array([1.0 + 1.0j])
        self.assertEqual(x.imag.item(), 1.0)
        self.assertEqual(x.real.item(), 1.0)

    def test_large_indices(self):
        x = tk.array([0, 1, 2])
        with self.assertRaises(ValueError):
            x[: 2**32]
        with self.assertRaises(ValueError):
            x[2**32]


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
