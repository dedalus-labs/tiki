# Copyright © 2024 Apple Inc.

import math
import os
import unittest

import tiki as tk
import tiki_tests
import numpy as np


class TestDouble(tiki_tests.TIKITestCase):
    def test_unary_ops(self):
        shape = (3, 3)
        x = tk.random.normal(shape=shape)

        if tk.default_device() == tk.gpu:
            with self.assertRaises(ValueError):
                x.astype(tk.float64)

        x_double = x.astype(tk.float64, stream=tk.cpu)

        ops = [
            tk.abs,
            tk.arccos,
            tk.arccosh,
            tk.arcsin,
            tk.arcsinh,
            tk.arctan,
            tk.arctanh,
            tk.ceil,
            tk.erf,
            tk.erfinv,
            tk.exp,
            tk.expm1,
            tk.floor,
            tk.log,
            tk.logical_not,
            tk.negative,
            tk.round,
            tk.sin,
            tk.sinh,
            tk.sqrt,
            tk.rsqrt,
            tk.tan,
            tk.tanh,
        ]
        for op in ops:
            if tk.default_device() == tk.gpu:
                with self.assertRaises(ValueError):
                    op(x_double)
                continue
            y = op(x)
            y_double = op(x_double)
            self.assertTrue(
                tk.allclose(y, y_double.astype(tk.float32, tk.cpu), equal_nan=True)
            )

    def test_binary_ops(self):
        shape = (3, 3)
        a = tk.random.normal(shape=shape)
        b = tk.random.normal(shape=shape)

        a_double = a.astype(tk.float64, stream=tk.cpu)
        b_double = b.astype(tk.float64, stream=tk.cpu)

        ops = [
            tk.add,
            tk.arctan2,
            tk.divide,
            tk.multiply,
            tk.subtract,
            tk.logical_and,
            tk.logical_or,
            tk.remainder,
            tk.maximum,
            tk.minimum,
            tk.power,
            tk.equal,
            tk.greater,
            tk.greater_equal,
            tk.less,
            tk.less_equal,
            tk.not_equal,
            tk.logaddexp,
        ]
        for op in ops:
            if tk.default_device() == tk.gpu:
                with self.assertRaises(ValueError):
                    op(a_double, b_double)
                continue
            y = op(a, b)
            y_double = op(a_double, b_double)
            self.assertTrue(
                tk.allclose(y, y_double.astype(tk.float32, tk.cpu), equal_nan=True)
            )

    def test_where(self):
        shape = (3, 3)
        cond = tk.random.uniform(shape=shape) > 0.5
        a = tk.random.normal(shape=shape)
        b = tk.random.normal(shape=shape)

        a_double = a.astype(tk.float64, stream=tk.cpu)
        b_double = b.astype(tk.float64, stream=tk.cpu)

        if tk.default_device() == tk.gpu:
            with self.assertRaises(ValueError):
                tk.where(cond, a_double, b_double)
            return
        y = tk.where(cond, a, b)
        y_double = tk.where(cond, a_double, b_double)
        self.assertTrue(tk.allclose(y, y_double.astype(tk.float32, tk.cpu)))

    def test_reductions(self):
        shape = (32, 32)
        a = tk.random.normal(shape=shape)
        a_double = a.astype(tk.float64, stream=tk.cpu)

        axes = [0, 1, (0, 1)]
        ops = [tk.sum, tk.prod, tk.min, tk.max, tk.any, tk.all]

        for op in ops:
            for ax in axes:
                if tk.default_device() == tk.gpu:
                    with self.assertRaises(ValueError):
                        op(a_double, axis=ax)
                    continue
                y = op(a)
                y_double = op(a_double)
                self.assertTrue(tk.allclose(y, y_double.astype(tk.float32, tk.cpu)))

    def test_get_and_set_item(self):
        shape = (3, 3)
        a = tk.random.normal(shape=shape)
        b = tk.random.normal(shape=(2,))
        a_double = a.astype(tk.float64, stream=tk.cpu)
        b_double = b.astype(tk.float64, stream=tk.cpu)
        idx_i = tk.array([0, 2])
        idx_j = tk.array([0, 2])

        if tk.default_device() == tk.gpu:
            with self.assertRaises(ValueError):
                a_double[idx_i, idx_j]
        else:
            y = a[idx_i, idx_j]
            y_double = a_double[idx_i, idx_j]
            self.assertTrue(tk.allclose(y, y_double.astype(tk.float32, tk.cpu)))

        if tk.default_device() == tk.gpu:
            with self.assertRaises(ValueError):
                a_double[idx_i, idx_j] = b_double
        else:
            a[idx_i, idx_j] = b
            a_double[idx_i, idx_j] = b_double
            self.assertTrue(tk.allclose(a, a_double.astype(tk.float32, tk.cpu)))

    def test_gemm(self):
        shape = (8, 8)
        a = tk.random.normal(shape=shape)
        b = tk.random.normal(shape=shape)

        a_double = a.astype(tk.float64, stream=tk.cpu)
        b_double = b.astype(tk.float64, stream=tk.cpu)

        if tk.default_device() == tk.gpu:
            with self.assertRaises(ValueError):
                a_double @ b_double
            return
        y = a @ b
        y_double = a_double @ b_double
        self.assertTrue(
            tk.allclose(y, y_double.astype(tk.float32, tk.cpu), equal_nan=True)
        )

    def test_type_promotion(self):
        import tiki as tk

        a = tk.array([4, 8], tk.float64)
        b = tk.array([4, 8], tk.int32)

        with tk.stream(tk.cpu):
            c = a + b
            self.assertEqual(c.dtype, tk.float64)

    def test_lapack(self):
        with tk.stream(tk.cpu):
            # QRF
            A = tk.array([[2.0, 3.0], [1.0, 2.0]], dtype=tk.float64)
            Q, R = tk.linalg.qr(A)
            out = Q @ R
            self.assertTrue(tk.allclose(out, A))
            out = Q.T @ Q
            self.assertTrue(tk.allclose(out, tk.eye(2)))
            self.assertTrue(tk.allclose(tk.tril(R, -1), tk.zeros_like(R)))
            self.assertEqual(Q.dtype, tk.float64)
            self.assertEqual(R.dtype, tk.float64)

            # SVD
            A = tk.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tk.float64
            )
            U, S, Vt = tk.linalg.svd(A)
            self.assertTrue(tk.allclose(U[:, : len(S)] @ tk.diag(S) @ Vt, A))

            # Inverse
            A = tk.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype=tk.float64)
            A_inv = tk.linalg.inv(A)
            self.assertTrue(tk.allclose(A @ A_inv, tk.eye(A.shape[0])))

            # Tri inv
            A = tk.array([[1, 0, 0], [6, -5, 0], [-9, 8, 7]], dtype=tk.float64)
            B = tk.array([[7, 0, 0], [3, -2, 0], [1, 8, 3]], dtype=tk.float64)
            AB = tk.stack([A, B])
            invs = tk.linalg.tri_inv(AB, upper=False)
            for M, M_inv in zip(AB, invs):
                self.assertTrue(tk.allclose(M @ M_inv, tk.eye(M.shape[0])))

            # Cholesky
            sqrtA = tk.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=tk.float64
            )
            A = sqrtA.T @ sqrtA / 81
            L = tk.linalg.cholesky(A)
            U = tk.linalg.cholesky(A, upper=True)
            self.assertTrue(tk.allclose(L @ L.T, A))
            self.assertTrue(tk.allclose(U.T @ U, A))

            # Psueod inverse
            A = tk.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype=tk.float64)
            A_plus = tk.linalg.pinv(A)
            self.assertTrue(tk.allclose(A @ A_plus @ A, A))

            # Eigh
            def check_eigs_and_vecs(A_np, kwargs={}):
                A = tk.array(A_np, dtype=tk.float64)
                eig_vals, eig_vecs = tk.linalg.eigh(A, **kwargs)
                eig_vals_np, _ = np.linalg.eigh(A_np, **kwargs)
                self.assertTrue(np.allclose(eig_vals, eig_vals_np))
                self.assertTrue(
                    tk.allclose(A @ eig_vecs, eig_vals[..., None, :] * eig_vecs)
                )

                eig_vals_only = tk.linalg.eigvalsh(A, **kwargs)
                self.assertTrue(tk.allclose(eig_vals, eig_vals_only))

            # Test a simple 2x2 symmetric matrix
            A_np = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
            check_eigs_and_vecs(A_np)

            # Test a larger random symmetric matrix
            n = 5
            np.random.seed(1)
            A_np = np.random.randn(n, n).astype(np.float64)
            A_np = (A_np + A_np.T) / 2
            check_eigs_and_vecs(A_np)

            # Test with upper triangle
            check_eigs_and_vecs(A_np, {"UPLO": "U"})

            # LU factorization
            # Test 3x3 matrix
            a = tk.array(
                [[3.0, 1.0, 2.0], [1.0, 8.0, 6.0], [9.0, 2.0, 5.0]], dtype=tk.float64
            )
            P, L, U = tk.linalg.lu(a)
            self.assertTrue(tk.allclose(L[P, :] @ U, a))

            # Solve triangular
            # Test lower triangular matrix
            a = tk.array(
                [[4.0, 0.0, 0.0], [2.0, 3.0, 0.0], [1.0, -2.0, 5.0]], dtype=tk.float64
            )
            b = tk.array([8.0, 14.0, 3.0], dtype=tk.float64)

            result = tk.linalg.solve_triangular(a, b, upper=False)
            expected = np.linalg.solve(np.array(a), np.array(b))
            self.assertTrue(np.allclose(result, expected))

            # Test upper triangular matrix
            a = tk.array(
                [[3.0, 2.0, 1.0], [0.0, 5.0, 4.0], [0.0, 0.0, 6.0]], dtype=tk.float64
            )
            b = tk.array([13.0, 33.0, 18.0], dtype=tk.float64)

            result = tk.linalg.solve_triangular(a, b, upper=True)
            expected = np.linalg.solve(np.array(a), np.array(b))
            self.assertTrue(np.allclose(result, expected))

    def test_conversion(self):
        a = tk.array([1.0, 2.0], tk.float64)
        b = np.array(a)
        self.assertTrue(np.array_equal(a, b))

        a = tk.array([1.0, 2.0], tk.float64)
        b = a.tolist()
        self.assertEqual(b, [1.0, 2.0])

    def test_python_float_keeps_double_precision(self):
        with tk.stream(tk.cpu):
            # https://github.com/ml-explore/mlx/issues/4160
            for v in (0.37, 0.1, math.pi):
                self.assertEqual(tk.full((3,), v, dtype=tk.float64).tolist(), [v] * 3)

            # https://github.com/ml-explore/mlx/issues/4159
            a = tk.array([1.0], dtype=tk.float64)
            for v in (0.1, 1e-4, math.pi):
                out = a * v
                self.assertEqual(out.dtype, tk.float64)
                self.assertEqual(out.tolist(), [v])

            # values outside the float32 range used to saturate to inf or zero
            self.assertEqual((a * 1e300).tolist(), [1e300])
            self.assertEqual((a * 1e-300).tolist(), [1e-300])

            # every op that pairs a scalar with an array goes through the same
            # conversion
            zero = tk.array([0.0], dtype=tk.float64)
            self.assertEqual(tk.maximum(zero, 0.1).tolist(), [0.1])
            self.assertEqual(tk.minimum(a, 0.1).tolist(), [0.1])
            self.assertEqual(tk.clip(a, None, 0.1).tolist(), [0.1])
            self.assertEqual(tk.where(tk.array([False]), zero, 0.1).tolist(), [0.1])
            self.assertEqual(
                tk.pad(a, 1, constant_values=0.1).tolist(), [0.1, 1.0, 0.1]
            )

            # the python float is still weak: it does not widen the array
            for dtype in (tk.float16, tk.bfloat16, tk.float32):
                self.assertEqual((tk.array([1.0], dtype=dtype) * 0.1).dtype, dtype)
            self.assertEqual((tk.array([1], dtype=tk.int32) * 0.1).dtype, tk.float32)

            # pad() keeps returning the input's dtype for every input type
            for dtype in (tk.int32, tk.float16, tk.bfloat16, tk.float32):
                padded = tk.pad(tk.array([1.0], dtype=dtype), 1, constant_values=0.5)
                self.assertEqual(padded.dtype, dtype)

    def test_linspace(self):
        with tk.stream(tk.cpu):
            vals = tk.linspace(0, math.pi, 2, dtype=tk.float64)
            self.assertEqual(vals.tolist()[1], math.pi)

            vals = tk.linspace(0, math.pi, 4, endpoint=False, dtype=tk.float64)
            self.assertEqual(vals.dtype, tk.float64)
            self.assertTrue(
                np.allclose(vals.tolist(), np.linspace(0, math.pi, 4, endpoint=False))
            )


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
