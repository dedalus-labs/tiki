# Copyright © 2023 Apple Inc.

import math
import unittest
from itertools import permutations

import numpy as np
import tiki as tk
import tiki_tests

try:
    import torch

    has_torch = True
except ImportError as e:
    has_torch = False

try:
    import ml_dtypes

    has_ml_dtypes = True
except ImportError:
    has_ml_dtypes = False


class TestBF16(tiki_tests.TIKITestCase):
    def __test_ops(
        self,
        ref_op,  # Function that outputs array_like
        tiki_op,  # Function that outputs array_like
        np_args,  # Numpy arguments
        ref_transform=lambda x: x,
        tiki_transform=lambda x: tk.array(x),
        atol=1e-5,
    ):
        ref_args = map(ref_transform, np_args)
        tiki_args = map(tiki_transform, np_args)

        r_ref = ref_op(*ref_args)
        r_tiki = tiki_op(*tiki_args)

        self.assertTrue(np.allclose(r_tiki, r_ref, atol=atol))

    def __default_test(
        self,
        op,
        np_args,
        simple_transform=lambda x: x,
        atol_np=1e-3,
        atol_torch=1e-5,
        np_kwargs=dict(),
        tiki_kwargs=dict(),
        torch_kwargs=dict(),
        torch_op=None,
    ):
        with self.subTest(reference="numpy"):

            def np_transform(x):
                x_mx_bf16 = tk.array(x).astype(tk.bfloat16)
                x_mx_fp32 = x_mx_bf16.astype(tk.float32)
                return np.asarray(x_mx_fp32)

            def tiki_fn(*args):
                out_bf16 = getattr(tk, op)(*args, **tiki_kwargs)
                return np.asarray(out_bf16.astype(tk.float32))

            def np_fn(*args):
                out_fp32 = getattr(np, op)(*args, **np_kwargs)
                return np_transform(out_fp32)

            ref_op = np_fn
            tiki_op = tiki_fn

            ref_transform = lambda x: simple_transform(np_transform(x))
            tiki_transform = lambda x: simple_transform(tk.array(x).astype(tk.bfloat16))

            self.__test_ops(
                ref_op,
                tiki_op,
                np_args,
                ref_transform=ref_transform,
                tiki_transform=tiki_transform,
                atol=atol_np,
            )

        if has_torch:
            with self.subTest(reference="torch"):
                torch_op = op if torch_op is None else torch_op

                def torch_fn(*args):
                    out_bf16 = getattr(torch, torch_op)(*args, **torch_kwargs)
                    return out_bf16.to(torch.float32).numpy()

                ref_op = torch_fn
                ref_transform = lambda x: simple_transform(
                    torch.from_numpy(x).to(torch.bfloat16)
                )
                self.__test_ops(
                    ref_op,
                    tiki_op,
                    np_args,
                    ref_transform=ref_transform,
                    tiki_transform=tiki_transform,
                    atol=atol_torch,
                )

    def test_unary_ops(self):
        x = np.random.rand(18, 28, 38)
        for op in ["abs", "exp", "log", "square", "sqrt"]:
            with self.subTest(op=op):
                np_args = (x.astype(np.float32),)
                self.__default_test(op, np_args)

    def test_binary_ops(self):
        x = np.random.rand(18, 28, 38)
        y = np.random.rand(18, 28, 38)
        for op in ["add", "subtract", "multiply", "divide", "maximum", "minimum"]:
            with self.subTest(op=op):
                np_args = (
                    x.astype(np.float32),
                    y.astype(np.float32),
                )
                self.__default_test(op, np_args, simple_transform=lambda x: x)
                self.__default_test(op, np_args, simple_transform=lambda x: x[:1])
                self.__default_test(op, np_args, simple_transform=lambda x: x[:, :1])

    def test_reduction_ops(self):
        x = np.random.rand(18, 28, 38).astype(np.float32)

        for op in ("min", "max"):
            with self.subTest(op=op):
                for axes in (0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)):
                    with self.subTest(axes=axes):
                        np_args = (x.astype(np.float32),)
                        self.__default_test(
                            op,
                            np_args,
                            np_kwargs={"axis": axes},
                            tiki_kwargs={"axis": axes},
                            torch_kwargs={"dim": axes},
                            torch_op="a" + op,
                        )

    def test_arithmetic_reduction_ops(self):
        cases = {
            "sum": np.ones((4096, 4), dtype=np.float32),
            "prod": np.full((128, 4), 1.0078125, dtype=np.float32),
        }
        for op, values in cases.items():
            with self.subTest(op=op):
                x = tk.array(values, dtype=tk.bfloat16)
                expected = tk.array(
                    getattr(np, op)(values, axis=0, dtype=np.float32),
                    dtype=tk.bfloat16,
                )
                actual = getattr(tk, op)(x, axis=0, stream=tk.cpu)
                self.assertEqual(actual.tolist(), expected.tolist())

    def test_arg_reduction_ops(self):
        data = np.random.rand(10, 12, 13).astype(np.float32)
        x = tk.array(data).astype(tk.bfloat16)
        data = np.asarray(x.astype(tk.float32))

        for op in ["argmin", "argmax"]:
            for axis in range(3):
                for kd in [True, False]:
                    a = getattr(tk, op)(x, axis, kd)
                    b = getattr(np, op)(data, axis, keepdims=kd)
                    a = a.astype(tk.float32)
                    self.assertEqual(a.tolist(), b.tolist())

        for op in ["argmin", "argmax"]:
            a = getattr(tk, op)(x, keepdims=True)
            b = getattr(np, op)(data, keepdims=True)
            a = a.astype(tk.float32)
            self.assertEqual(a.tolist(), b.tolist())
            a = getattr(tk, op)(x)
            b = getattr(np, op)(data)
            a = a.astype(tk.float32)
            self.assertEqual(a.item(), b)

    def test_blas_ops(self):
        if tk.default_device() != tk.gpu:
            return

        def test_blas(shape_x, shape_y):
            np.random.seed(42)
            with self.subTest(shape_x=shape_x, shape_y=shape_y):
                x = np.random.normal(0.0, 1.0 / shape_x[-1], size=shape_x)
                y = np.random.normal(0.0, 1.0 / shape_x[-1], size=shape_y)

                np_args = (
                    x.astype(np.float32),
                    y.astype(np.float32),
                )
                op = "matmul"

                self.__default_test(op, np_args, atol_np=1e-3, atol_torch=1e-3)

        for shape_x, shape_y in [
            [(32, 32), (32, 32)],
            [(23, 57), (57, 1)],
            [(1, 3), (3, 128)],
            [(8, 128, 768), (768, 16)],
        ]:
            test_blas(shape_x, shape_y)

    @unittest.skipIf(not has_torch, "requires PyTorch")
    def test_conversion(self):
        a_torch = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        a_mx = tk.array(a_torch)
        expected = tk.array([1.0, 2.0, 3.0], tk.bfloat16)
        self.assertEqual(a_mx.dtype, tk.bfloat16)
        self.assertTrue(tk.array_equal(a_mx, expected))

    @unittest.skipIf(not has_ml_dtypes, "requires ml_dtypes")
    def test_conversion_ml_dtypes(self):
        x_scalar = np.array(1.5, dtype=ml_dtypes.bfloat16)
        a_scalar = tk.array(x_scalar)
        self.assertEqual(a_scalar.dtype, tk.bfloat16)
        self.assertEqual(a_scalar.shape, ())
        self.assertEqual(a_scalar.item(), 1.5)

        data = [1.5, 2.5, 3.5]
        x_vector = np.array(data, dtype=ml_dtypes.bfloat16)
        a_vector = tk.array(x_vector)
        expected = tk.array(data, dtype=tk.bfloat16)
        self.assertEqual(a_vector.dtype, tk.bfloat16)
        self.assertEqual(a_vector.shape, (3,))
        self.assertTrue(tk.array_equal(a_vector, expected))

        a_cast = tk.array(x_scalar, dtype=tk.float32)
        self.assertEqual(a_cast.dtype, tk.float32)
        self.assertEqual(a_cast.item(), 1.5)

        a_asarray = tk.asarray(x_vector)
        self.assertEqual(a_asarray.dtype, tk.bfloat16)
        self.assertEqual(a_asarray.shape, (3,))
        self.assertTrue(tk.array_equal(a_asarray, expected))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
