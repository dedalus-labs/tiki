# Copyright © 2023 Apple Inc.

import itertools
import math
import unittest

import numpy as np
import tiki as tk
import tiki_tests


class TestLinalg(tiki_tests.TIKITestCase):
    def test_norm(self):
        vector_ords = [None, 0.5, 0, 1, 2, 3, -1, float("inf"), -float("inf")]
        matrix_ords = [None, "fro", "nuc", -1, 1, -2, 2, float("inf"), -float("inf")]

        for shape in [(3,), (2, 3), (2, 3, 3)]:
            x_mx = tk.arange(1, math.prod(shape) + 1, dtype=tk.float32).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1, dtype=np.float32).reshape(shape)
            # Test when at least one axis is provided
            for num_axes in range(1, len(shape)):
                if num_axes == 1:
                    ords = vector_ords
                else:
                    ords = matrix_ords
                for axis in itertools.combinations(range(len(shape)), num_axes):
                    for keepdims in [True, False]:
                        for o in ords:
                            stream = (
                                tk.cpu if o in ["nuc", -2, 2] else tk.default_device()
                            )
                            out_np = np.linalg.norm(
                                x_np, ord=o, axis=axis, keepdims=keepdims
                            )
                            out_mx = tk.linalg.norm(
                                x_mx, ord=o, axis=axis, keepdims=keepdims, stream=stream
                            )
                            with self.subTest(
                                shape=shape, ord=o, axis=axis, keepdims=keepdims
                            ):
                                self.assertEqual(out_mx.shape, out_np.shape)
                                self.assertTrue(
                                    np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)
                                )

        # Test only ord provided
        for shape in [(3,), (2, 3)]:
            x_mx = tk.arange(1, math.prod(shape) + 1).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1).reshape(shape)
            for o in [None, 1, -1, float("inf"), -float("inf")]:
                for keepdims in [True, False]:
                    out_np = np.linalg.norm(x_np, ord=o, keepdims=keepdims)
                    out_mx = tk.linalg.norm(x_mx, ord=o, keepdims=keepdims)
                    with self.subTest(shape=shape, ord=o, keepdims=keepdims):
                        self.assertEqual(out_mx.shape, out_np.shape)
                        self.assertTrue(
                            np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)
                        )

        # Test no ord and no axis provided
        for shape in [(3,), (2, 3), (2, 3, 3)]:
            x_mx = tk.arange(1, math.prod(shape) + 1).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1).reshape(shape)
            for keepdims in [True, False]:
                out_np = np.linalg.norm(x_np, keepdims=keepdims)
                out_mx = tk.linalg.norm(x_mx, keepdims=keepdims)
                with self.subTest(shape=shape, keepdims=keepdims):
                    self.assertEqual(out_mx.shape, out_np.shape)
                    self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6))

        # tests for negative indexing: -1/1/inf/-inf/
        norms = [-1, 1, -float("inf"), float("inf")]
        for shape in [(3, 3), (2, 3, 3), (2, 3, 3, 3)]:
            x_mx = tk.arange(1, math.prod(shape) + 1, dtype=tk.float32).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1, dtype=np.float32).reshape(shape)
            neg_indices = [-i for i in range(1, x_np.ndim + 1)]
            neg_axes = [list(p) for p in itertools.permutations(neg_indices, 2)]
            for ord in norms:
                for axes in neg_axes:
                    out_np = np.linalg.norm(
                        x_np,
                        ord=ord,
                        axis=tuple(axes),
                    )
                    out_mx = tk.linalg.norm(x_mx, ord=ord, axis=axes)
                    with self.subTest(ord=ord, axes=axes):
                        self.assertTrue(
                            np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)
                        )

    def test_complex_norm(self):
        for shape in [(3,), (2, 3), (2, 3, 3)]:
            x_np = np.random.uniform(size=shape).astype(
                np.float32
            ) + 1j * np.random.uniform(size=shape).astype(np.float32)
            x_mx = tk.array(x_np)
            out_np = np.linalg.norm(x_np)
            out_mx = tk.linalg.norm(x_mx)
            with self.subTest(shape=shape):
                self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6))
            for num_axes in range(1, len(shape)):
                for axis in itertools.combinations(range(len(shape)), num_axes):
                    out_np = np.linalg.norm(x_np, axis=axis)
                    out_mx = tk.linalg.norm(x_mx, axis=axis)
                    with self.subTest(shape=shape, axis=axis):
                        self.assertTrue(
                            np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)
                        )

        x_np = np.random.uniform(size=(4, 4)).astype(
            np.float32
        ) + 1j * np.random.uniform(size=(4, 4)).astype(np.float32)
        x_mx = tk.array(x_np)
        out_np = np.linalg.norm(x_np, ord="fro")
        out_mx = tk.linalg.norm(x_mx, ord="fro")
        self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6))

    def test_qr_factorization(self):
        with self.assertRaises(ValueError):
            tk.linalg.qr(tk.array(0.0))

        with self.assertRaises(ValueError):
            tk.linalg.qr(tk.array([0.0, 1.0]))

        with self.assertRaises(ValueError):
            tk.linalg.qr(tk.array([[0, 1], [1, 0]]))

        A = tk.array([[2.0, 3.0], [1.0, 2.0]])
        Q, R = tk.linalg.qr(A, stream=tk.cpu)
        out = Q @ R
        self.assertTrue(tk.allclose(out, A))
        out = Q.T @ Q
        self.assertTrue(tk.allclose(out, tk.eye(2), rtol=1e-5, atol=1e-7))
        self.assertTrue(tk.allclose(tk.tril(R, -1), tk.zeros_like(R)))
        self.assertEqual(Q.dtype, tk.float32)
        self.assertEqual(R.dtype, tk.float32)

        # Multiple matrices
        B = tk.array([[-1.0, 2.0], [-4.0, 1.0]])
        A = tk.stack([A, B])
        Q, R = tk.linalg.qr(A, stream=tk.cpu)
        for a, q, r in zip(A, Q, R):
            out = q @ r
            self.assertTrue(tk.allclose(out, a))
            out = q.T @ q
            self.assertTrue(tk.allclose(out, tk.eye(2), rtol=1e-5, atol=1e-7))
            self.assertTrue(tk.allclose(tk.tril(r, -1), tk.zeros_like(r)))

        # Non square matrices
        for shape in [(4, 8), (8, 4)]:
            A = tk.random.uniform(shape=shape)
            Q, R = tk.linalg.qr(A, stream=tk.cpu)
            out = Q @ R
            self.assertTrue(tk.allclose(out, A, rtol=1e-4, atol=1e-6))
            out = Q.T @ Q
            self.assertTrue(
                tk.allclose(out, tk.eye(min(A.shape)), rtol=1e-4, atol=1e-6)
            )

        # Zero-size inputs. Both factors carry min(M, N) as a dimension, so
        # both are empty whichever dimension is zero.
        for shape in [(0, 0), (3, 0, 0), (0, 4, 4), (5, 0), (0, 5)]:
            A_np = np.zeros(shape, dtype=np.float32)
            Q, R = tk.linalg.qr(tk.array(A_np), stream=tk.cpu)
            tk.eval(Q, R)
            Q_np, R_np = np.linalg.qr(A_np)
            self.assertEqual(Q.shape, Q_np.shape)
            self.assertEqual(R.shape, R_np.shape)

    def test_svd_decomposition(self):
        A = tk.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tk.float32)
        U, S, Vt = tk.linalg.svd(A, compute_uv=True, stream=tk.cpu)
        self.assertTrue(
            tk.allclose(U[:, : len(S)] @ tk.diag(S) @ Vt, A, rtol=1e-5, atol=1e-7)
        )

        S = tk.linalg.svd(A, compute_uv=False, stream=tk.cpu)
        self.assertTrue(
            tk.allclose(
                tk.linalg.norm(S), tk.linalg.norm(A, ord="fro"), rtol=1e-5, atol=1e-7
            )
        )

        # Multiple matrices
        B = A + 10.0
        AB = tk.stack([A, B])
        Us, Ss, Vts = tk.linalg.svd(AB, compute_uv=True, stream=tk.cpu)
        for M, U, S, Vt in zip([A, B], Us, Ss, Vts):
            self.assertTrue(
                tk.allclose(U[:, : len(S)] @ tk.diag(S) @ Vt, M, rtol=1e-5, atol=1e-7)
            )

        Ss = tk.linalg.svd(AB, compute_uv=False, stream=tk.cpu)
        for M, S in zip([A, B], Ss):
            self.assertTrue(
                tk.allclose(
                    tk.linalg.norm(S),
                    tk.linalg.norm(M, ord="fro"),
                    rtol=1e-5,
                    atol=1e-7,
                )
            )

        # Zero-size inputs. When only one of the dimensions is zero the
        # factors are not empty and hold the identity, like numpy.
        for shape in [(0, 4, 4), (3, 0, 0), (2, 5, 0), (2, 0, 5), (5, 0), (0, 5)]:
            a_np = np.zeros(shape, dtype=np.float32)
            U, S, Vt = tk.linalg.svd(tk.array(a_np), stream=tk.cpu)
            tk.eval(U, S, Vt)
            U_np, S_np, Vt_np = np.linalg.svd(a_np)
            self.assertEqual(U.shape, U_np.shape)
            self.assertEqual(S.shape, S_np.shape)
            self.assertEqual(Vt.shape, Vt_np.shape)
            self.assertTrue(np.array_equal(np.array(U), U_np))
            self.assertTrue(np.array_equal(np.array(Vt), Vt_np))

            S_only = tk.linalg.svd(tk.array(a_np), compute_uv=False, stream=tk.cpu)
            tk.eval(S_only)
            self.assertEqual(S_only.shape, S_np.shape)

        # Test float64 - use CPU stream since float64 is not supported on GPU
        with tk.stream(tk.cpu):
            A_f64 = tk.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tk.float64
            )
            U_f64, S_f64, Vt_f64 = tk.linalg.svd(A_f64, compute_uv=True)
            tk.eval(U_f64, S_f64, Vt_f64)
            self.assertTrue(
                tk.allclose(
                    U_f64[:, : len(S_f64)] @ tk.diag(S_f64) @ Vt_f64,
                    A_f64,
                    rtol=1e-5,
                    atol=1e-7,
                )
            )
            self.assertEqual(S_f64.dtype, tk.float64)

        # Test complex64 - use CPU stream since complex64 is not supported on GPU
        with tk.stream(tk.cpu):
            A_c64 = tk.array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], dtype=tk.complex64
            )
            U_c64, S_c64, Vt_c64 = tk.linalg.svd(A_c64, compute_uv=True)
            tk.eval(U_c64, S_c64, Vt_c64)
            self.assertTrue(
                tk.allclose(
                    U_c64[:, : len(S_c64)] @ tk.diag(S_c64) @ Vt_c64,
                    A_c64,
                    rtol=1e-5,
                    atol=1e-7,
                )
            )
            self.assertEqual(S_c64.dtype, tk.float32)
            self.assertEqual(U_c64.dtype, tk.complex64)
            self.assertEqual(Vt_c64.dtype, tk.complex64)

    def test_inverse(self):
        A = tk.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype=tk.float32)
        A_inv = tk.linalg.inv(A, stream=tk.cpu)
        self.assertTrue(tk.allclose(A @ A_inv, tk.eye(A.shape[0]), rtol=0, atol=1e-6))

        # Multiple matrices
        B = A - 100
        AB = tk.stack([A, B])
        invs = tk.linalg.inv(AB, stream=tk.cpu)
        for M, M_inv in zip(AB, invs):
            self.assertTrue(
                tk.allclose(M @ M_inv, tk.eye(M.shape[0]), rtol=0, atol=1e-5)
            )

    def test_tri_inverse(self):
        for upper in (False, True):
            A = tk.array([[1, 0, 0], [6, -5, 0], [-9, 8, 7]], dtype=tk.float32)
            B = tk.array([[7, 0, 0], [3, -2, 0], [1, 8, 3]], dtype=tk.float32)
            if upper:
                A = A.T
                B = B.T
            AB = tk.stack([A, B])
            invs = tk.linalg.tri_inv(AB, upper=upper, stream=tk.cpu)
            for M, M_inv in zip(AB, invs):
                self.assertTrue(
                    tk.allclose(M @ M_inv, tk.eye(M.shape[0]), rtol=0, atol=1e-5)
                )

        # Ensure that tri_inv will 0-out the supposedly 0 triangle
        x = tk.random.normal((2, 8, 8))
        y1 = tk.linalg.tri_inv(x, upper=True, stream=tk.cpu)
        y2 = tk.linalg.tri_inv(x, upper=False, stream=tk.cpu)
        self.assertTrue(tk.all(y1 == tk.triu(y1)))
        self.assertTrue(tk.all(y2 == tk.tril(y2)))

    def test_cholesky(self):
        sqrtA = tk.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=tk.float32
        )
        A = sqrtA.T @ sqrtA / 81
        L = tk.linalg.cholesky(A, stream=tk.cpu)
        U = tk.linalg.cholesky(A, upper=True, stream=tk.cpu)
        self.assertTrue(tk.allclose(L @ L.T, A, rtol=1e-5, atol=1e-7))
        self.assertTrue(tk.allclose(U.T @ U, A, rtol=1e-5, atol=1e-7))

        # Multiple matrices
        B = A + 1 / 9
        AB = tk.stack([A, B])
        Ls = tk.linalg.cholesky(AB, stream=tk.cpu)
        for M, L in zip(AB, Ls):
            self.assertTrue(tk.allclose(L @ L.T, M, rtol=1e-5, atol=1e-7))

        if tk.cuda.is_available():
            # The matrix above is singular, so its factor is undefined past the
            # rank boundary and the GPU need not match the CPU there. Use
            # positive definite inputs, sized to hit both cuSOLVER paths.
            tk.random.seed(7)
            for batch, n in [((), 3), ((2,), 2048), ((16,), 8)]:
                x = tk.random.normal(batch + (n, n))
                A = x @ tk.swapaxes(x, -1, -2) + n * tk.eye(n)
                for upper in [False, True]:
                    G = tk.linalg.cholesky(A, upper=upper, stream=tk.gpu)
                    C = tk.linalg.cholesky(A, upper=upper, stream=tk.cpu)
                    self.assertTrue(tk.allclose(G, C, rtol=1e-4, atol=1e-5))
                    tri = tk.tril(G, k=-1) if upper else tk.triu(G, k=1)
                    self.assertTrue(tk.all(tri == 0))

            # Empty and non contiguous inputs
            self.assertEqual(
                tk.linalg.cholesky(tk.zeros((0, 0)), stream=tk.gpu).shape, (0, 0)
            )
            At = tk.swapaxes(A, -1, -2)
            self.assertTrue(
                tk.allclose(
                    tk.linalg.cholesky(At, stream=tk.gpu),
                    tk.linalg.cholesky(At, stream=tk.cpu),
                    rtol=1e-4,
                    atol=1e-5,
                )
            )

    def test_pseudo_inverse(self):
        A = tk.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype=tk.float32)
        A_plus = tk.linalg.pinv(A, stream=tk.cpu)
        self.assertTrue(tk.allclose(A @ A_plus @ A, A, rtol=0, atol=1e-5))

        # Multiple matrices
        B = A - 100
        AB = tk.stack([A, B])
        pinvs = tk.linalg.pinv(AB, stream=tk.cpu)
        for M, M_plus in zip(AB, pinvs):
            self.assertTrue(tk.allclose(M @ M_plus @ M, M, rtol=0, atol=1e-3))

        # Test singular matrix
        A = tk.array([[4.0, 1.0], [4.0, 1.0]])
        A_plus = tk.linalg.pinv(A, stream=tk.cpu)
        self.assertTrue(tk.allclose(A @ A_plus @ A, A))

        # Zero-size inputs. The result takes the shape of the transposed input.
        for shape in [(0, 0), (0, 3), (3, 0), (0, 2, 2), (2, 0, 0), (0, 4, 3)]:
            A_np = np.zeros(shape, dtype=np.float32)
            A_plus = tk.linalg.pinv(tk.array(A_np), stream=tk.cpu)
            tk.eval(A_plus)
            self.assertEqual(A_plus.shape, np.linalg.pinv(A_np).shape)

    def test_cholesky_inv(self):
        tk.random.seed(7)

        sqrtA = tk.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=tk.float32
        )
        A = sqrtA.T @ sqrtA / 81

        N = 3
        A = tk.random.uniform(shape=(N, N))
        A = A @ A.T

        for upper in (False, True):
            L = tk.linalg.cholesky(A, upper=upper, stream=tk.cpu)
            A_inv = tk.linalg.cholesky_inv(L, upper=upper, stream=tk.cpu)
            self.assertTrue(tk.allclose(A @ A_inv, tk.eye(N), atol=1e-4))

        # Multiple matrices
        B = A + 1 / 9
        AB = tk.stack([A, B])
        Ls = tk.linalg.cholesky(AB, stream=tk.cpu)
        for upper in (False, True):
            Ls = tk.linalg.cholesky(AB, upper=upper, stream=tk.cpu)
            AB_inv = tk.linalg.cholesky_inv(Ls, upper=upper, stream=tk.cpu)
            for M, M_inv in zip(AB, AB_inv):
                self.assertTrue(tk.allclose(M @ M_inv, tk.eye(N), atol=1e-4))

    def test_cross_product(self):
        a = tk.array([1.0, 2.0, 3.0])
        b = tk.array([4.0, 5.0, 6.0])
        result = tk.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test with negative values
        a = tk.array([-1.0, -2.0, -3.0])
        b = tk.array([4.0, -5.0, 6.0])
        result = tk.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test with integer values
        a = tk.array([1, 2, 3])
        b = tk.array([4, 5, 6])
        result = tk.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test with 2D arrays and axis parameter
        a = tk.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tk.array([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
        result = tk.linalg.cross(a, b, axis=1)
        expected = np.cross(a, b, axis=1)
        self.assertTrue(np.allclose(result, expected))

        # Test with broadcast
        a = tk.random.uniform(shape=(2, 1, 3))
        b = tk.random.uniform(shape=(1, 2, 3))
        result = tk.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Type promotion
        a = tk.array([1.0, 2.0, 3.0])
        b = tk.array([4, 5, 6])
        result = tk.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test with incorrect vector size (should raise an exception)
        a = tk.array([1.0])
        b = tk.array([4.0])
        with self.assertRaises(ValueError):
            tk.linalg.cross(a, b)

    def test_eig(self):
        tols = {"atol": 1e-5, "rtol": 1e-5}

        def check_eigs_and_vecs(A_np, kwargs={}):
            A = tk.array(A_np)
            eig_vals, eig_vecs = tk.linalg.eig(A, stream=tk.cpu, **kwargs)
            self.assertTrue(
                tk.allclose(A @ eig_vecs, eig_vals[..., None, :] * eig_vecs, **tols)
            )
            eig_vals_only = tk.linalg.eigvals(A, stream=tk.cpu, **kwargs)
            self.assertTrue(tk.allclose(eig_vals, eig_vals_only, **tols))

        # Test a simple 2x2 matrix
        A_np = np.array([[1.0, 1.0], [3.0, 4.0]], dtype=np.float32)
        check_eigs_and_vecs(A_np)

        # Test complex eigenvalues
        A_np = np.array([[1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
        check_eigs_and_vecs(A_np)

        # Test a larger random symmetric matrix
        n = 5
        np.random.seed(1)
        A_np = np.random.randn(n, n).astype(np.float32)
        check_eigs_and_vecs(A_np)

        # Test with batched input
        A_np = np.random.randn(3, n, n).astype(np.float32)
        check_eigs_and_vecs(A_np)

        # Test float64 - use CPU stream since float64 is not supported on GPU
        with tk.stream(tk.cpu):
            A_np_f64 = np.array([[1.0, 1.0], [3.0, 4.0]], dtype=np.float64)
            A_f64 = tk.array(A_np_f64, dtype=tk.float64)
            eig_vals_f64, eig_vecs_f64 = tk.linalg.eig(A_f64)
            tk.eval(eig_vals_f64, eig_vecs_f64)
            self.assertTrue(
                tk.allclose(
                    A_f64 @ eig_vecs_f64,
                    eig_vals_f64[..., None, :] * eig_vecs_f64,
                    rtol=1e-5,
                    atol=1e-5,
                )
            )
            # Eigenvalues should be complex64 (output dtype)
            self.assertEqual(eig_vals_f64.dtype, tk.complex64)
            self.assertEqual(eig_vecs_f64.dtype, tk.complex64)

        # Test complex64 input - use CPU stream since complex64 is not supported on GPU
        with tk.stream(tk.cpu):
            A_np_c64 = np.array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], dtype=np.complex64
            )
            A_c64 = tk.array(A_np_c64, dtype=tk.complex64)
            eig_vals_c64, eig_vecs_c64 = tk.linalg.eig(A_c64)
            tk.eval(eig_vals_c64, eig_vecs_c64)
            self.assertTrue(
                tk.allclose(
                    A_c64 @ eig_vecs_c64,
                    eig_vals_c64[..., None, :] * eig_vecs_c64,
                    rtol=1e-5,
                    atol=1e-5,
                )
            )
            self.assertEqual(eig_vals_c64.dtype, tk.complex64)
            self.assertEqual(eig_vecs_c64.dtype, tk.complex64)

        # Zero-size inputs. The input is square, so both outputs are empty.
        for shape in [(0, 0), (3, 0, 0), (0, 4, 4)]:
            A_np = np.zeros(shape, dtype=np.float32)
            eig_vals, eig_vecs = tk.linalg.eig(tk.array(A_np), stream=tk.cpu)
            tk.eval(eig_vals, eig_vecs)
            vals_np, vecs_np = np.linalg.eig(A_np)
            self.assertEqual(eig_vals.shape, vals_np.shape)
            self.assertEqual(eig_vecs.shape, vecs_np.shape)

            vals_only = tk.linalg.eigvals(tk.array(A_np), stream=tk.cpu)
            tk.eval(vals_only)
            self.assertEqual(vals_only.shape, vals_np.shape)

        # Test error cases
        with self.assertRaises(ValueError):
            tk.linalg.eig(tk.array([1.0, 2.0]))  # 1D array

        with self.assertRaises(ValueError):
            tk.linalg.eig(
                tk.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )  # Non-square matrix

        with self.assertRaises(ValueError):
            tk.linalg.eigvals(tk.array([1.0, 2.0]))  # 1D array

        with self.assertRaises(ValueError):
            tk.linalg.eigvals(
                tk.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )  # Non-square matrix

    def test_eigh(self):
        tols = {"atol": 1e-5, "rtol": 1e-5}

        def check_eigs_and_vecs(A_np, kwargs={}):
            A = tk.array(A_np)
            eig_vals, eig_vecs = tk.linalg.eigh(A, stream=tk.cpu, **kwargs)
            eig_vals_np, _ = np.linalg.eigh(A_np, **kwargs)
            self.assertTrue(np.allclose(eig_vals, eig_vals_np, **tols))
            self.assertTrue(
                tk.allclose(A @ eig_vecs, eig_vals[..., None, :] * eig_vecs, **tols)
            )

            eig_vals_only = tk.linalg.eigvalsh(A, stream=tk.cpu, **kwargs)
            self.assertTrue(tk.allclose(eig_vals, eig_vals_only, **tols))

        # Test a simple 2x2 symmetric matrix
        A_np = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        check_eigs_and_vecs(A_np)

        # Test a larger random symmetric matrix
        n = 5
        np.random.seed(1)
        A_np = np.random.randn(n, n).astype(np.float32)
        A_np = (A_np + A_np.T) / 2
        check_eigs_and_vecs(A_np)

        # Test with upper triangle
        check_eigs_and_vecs(A_np, {"UPLO": "U"})

        # Test with batched input
        A_np = np.random.randn(3, n, n).astype(np.float32)
        A_np = (A_np + np.transpose(A_np, (0, 2, 1))) / 2
        check_eigs_and_vecs(A_np)

        # Test with complex inputs
        A_np = (
            np.random.randn(8, 8, 2).astype(np.float32).view(np.complex64).squeeze(-1)
        )
        A_np = A_np + A_np.T.conj()
        check_eigs_and_vecs(A_np)

        # UPLO picks the triangle like numpy; only observable when the two
        # triangles disagree
        A_np = np.array([[1.0, 999.0], [2.0, 3.0]], dtype=np.float32)
        for uplo in ("L", "U"):
            w = tk.linalg.eigvalsh(tk.array(A_np), UPLO=uplo, stream=tk.cpu)
            w_np = np.linalg.eigvalsh(A_np, UPLO=uplo)
            self.assertTrue(np.allclose(w, w_np, atol=1e-5))

        # Zero-size inputs
        for shape in [(0, 4, 4), (3, 0, 0), (0, 0)]:
            a_np = np.zeros(shape, dtype=np.float32)
            w, v = tk.linalg.eigh(tk.array(a_np), stream=tk.cpu)
            tk.eval(w, v)
            w_np, v_np = np.linalg.eigh(a_np)
            self.assertEqual(w.shape, w_np.shape)
            self.assertEqual(v.shape, v_np.shape)
            self.assertTrue(np.array_equal(np.array(w), w_np))
            self.assertTrue(np.array_equal(np.array(v), v_np))

            w_only = tk.linalg.eigvalsh(tk.array(a_np), stream=tk.cpu)
            tk.eval(w_only)
            self.assertEqual(w_only.shape, w_np.shape)

        # Test error cases
        with self.assertRaises(ValueError):
            tk.linalg.eigh(tk.array([1.0, 2.0]))  # 1D array

        with self.assertRaises(ValueError):
            tk.linalg.eigh(
                tk.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )  # Non-square matrix

        with self.assertRaises(ValueError):
            tk.linalg.eigvalsh(tk.array([1.0, 2.0]))  # 1D array

        with self.assertRaises(ValueError):
            tk.linalg.eigvalsh(
                tk.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )  # Non-square matrix

    def test_lu(self):
        with self.assertRaises(ValueError):
            tk.linalg.lu(tk.array(0.0), stream=tk.cpu)

        with self.assertRaises(ValueError):
            tk.linalg.lu(tk.array([0.0, 1.0]), stream=tk.cpu)

        with self.assertRaises(ValueError):
            tk.linalg.lu(tk.array([[0, 1], [1, 0]]), stream=tk.cpu)

        # Test 3x3 matrix
        a = tk.array([[3.0, 1.0, 2.0], [1.0, 8.0, 6.0], [9.0, 2.0, 5.0]])
        P, L, U = tk.linalg.lu(a, stream=tk.cpu)
        self.assertTrue(tk.allclose(L[P, :] @ U, a))

        # Test batch dimension
        a = tk.broadcast_to(a, (5, 5, 3, 3))
        P, L, U = tk.linalg.lu(a, stream=tk.cpu)
        L = tk.take_along_axis(L, P[..., None], axis=-2)
        self.assertTrue(tk.allclose(L @ U, a))

        # Test non-square matrix
        a = tk.array([[3.0, 1.0, 2.0], [1.0, 8.0, 6.0]])
        P, L, U = tk.linalg.lu(a, stream=tk.cpu)
        self.assertTrue(tk.allclose(L[P, :] @ U, a))

        a = tk.array([[3.0, 1.0], [1.0, 8.0], [9.0, 2.0]])
        P, L, U = tk.linalg.lu(a, stream=tk.cpu)
        self.assertTrue(tk.allclose(L[P, :] @ U, a))

        # Test singular matrix (should not throw)
        a = tk.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
        P, L, U = tk.linalg.lu(a, stream=tk.cpu)
        L_permuted = tk.take_along_axis(L, P[..., None], axis=-2)
        self.assertTrue(tk.allclose(L_permuted @ U, a))

    def test_lu_factor(self):
        tk.random.seed(7)

        # Test 3x3 matrix
        a = tk.random.uniform(shape=(5, 5))
        LU, pivots = tk.linalg.lu_factor(a, stream=tk.cpu)
        n = a.shape[-1]

        pivots = pivots.tolist()
        perm = list(range(n))
        for i in range(len(pivots)):
            perm[i], perm[pivots[i]] = perm[pivots[i]], perm[i]

        L = tk.add(tk.tril(LU, k=-1), tk.eye(n))
        U = tk.triu(LU)
        self.assertTrue(tk.allclose(L @ U, a[perm, :]))

    def test_solve(self):
        tk.random.seed(7)

        # Test 3x3 matrix with 1D rhs
        a = tk.array([[3.0, 1.0, 2.0], [1.0, 8.0, 6.0], [9.0, 2.0, 5.0]])
        b = tk.array([11.0, 35.0, 28.0])

        result = tk.linalg.solve(a, b, stream=tk.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test symmetric positive-definite matrix
        N = 5
        a = tk.random.uniform(shape=(N, N))
        a = tk.matmul(a, a.T) + N * tk.eye(N)
        b = tk.random.uniform(shape=(N, 1))

        result = tk.linalg.solve(a, b, stream=tk.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test batch dimension
        a = tk.random.uniform(shape=(5, 5, 4, 4))
        b = tk.random.uniform(shape=(5, 5, 4, 1))

        result = tk.linalg.solve(a, b, stream=tk.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected, atol=1e-5))

        # Test large matrix
        N = 1000
        a = tk.random.uniform(shape=(N, N))
        b = tk.random.uniform(shape=(N, 1))

        result = tk.linalg.solve(a, b, stream=tk.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

        # Test multi-column rhs
        a = tk.random.uniform(shape=(5, 5))
        b = tk.random.uniform(shape=(5, 8))

        result = tk.linalg.solve(a, b, stream=tk.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test batched multi-column rhs
        a = tk.broadcast_to(a, (3, 2, 5, 5))
        b = tk.broadcast_to(b, (3, 1, 5, 8))

        result = tk.linalg.solve(a, b, stream=tk.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected, rtol=1e-5, atol=1e-5))

    def test_solve_triangular(self):
        # Test lower triangular matrix
        a = tk.array([[4.0, 0.0, 0.0], [2.0, 3.0, 0.0], [1.0, -2.0, 5.0]])
        b = tk.array([8.0, 14.0, 3.0])

        result = tk.linalg.solve_triangular(a, b, upper=False, stream=tk.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test upper triangular matrix
        a = tk.array([[3.0, 2.0, 1.0], [0.0, 5.0, 4.0], [0.0, 0.0, 6.0]])
        b = tk.array([13.0, 33.0, 18.0])

        result = tk.linalg.solve_triangular(a, b, upper=True, stream=tk.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test batch multi-column rhs
        a = tk.broadcast_to(a, (3, 4, 3, 3))
        b = tk.broadcast_to(tk.expand_dims(b, -1), (3, 4, 3, 8))

        result = tk.linalg.solve_triangular(a, b, upper=True, stream=tk.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

    def test_det(self):
        # 1x1 fast path
        A = tk.array([[5.0]])
        self.assertTrue(np.allclose(tk.linalg.det(A, stream=tk.cpu), 5.0))

        # 2x2 fast path
        A = tk.array([[1.0, 2.0], [3.0, 4.0]])
        d = tk.linalg.det(A, stream=tk.cpu)
        self.assertTrue(np.allclose(d, -2.0))

        # 3x3 fast path
        A = tk.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
        d = tk.linalg.det(A, stream=tk.cpu)
        expected = np.linalg.det(np.array(A))
        self.assertTrue(np.allclose(d, expected, atol=1e-5))

        # 4x4 LU path: compare with numpy
        np.random.seed(42)
        A_np = np.random.randn(4, 4).astype(np.float32)
        A_mx = tk.array(A_np)
        d_mx = tk.linalg.det(A_mx, stream=tk.cpu)
        d_np = np.linalg.det(A_np)
        self.assertTrue(np.allclose(d_mx, d_np, atol=1e-4))

        # 5x5 LU path
        A_np = np.random.randn(5, 5).astype(np.float32)
        A_mx = tk.array(A_np)
        d_mx = tk.linalg.det(A_mx, stream=tk.cpu)
        d_np = np.linalg.det(A_np)
        self.assertTrue(np.allclose(d_mx, d_np, atol=1e-4))

        # Identity matrix
        A = tk.eye(5)
        self.assertTrue(np.allclose(tk.linalg.det(A, stream=tk.cpu), 1.0))

        # Batched: (3, 4, 4)
        A_np = np.random.randn(3, 4, 4).astype(np.float32)
        A_mx = tk.array(A_np)
        d_mx = tk.linalg.det(A_mx, stream=tk.cpu)
        d_np = np.linalg.det(A_np)
        self.assertTrue(np.allclose(d_mx, d_np, atol=1e-4))

        # Multi-batch: (2, 3, 3, 3)
        A_np = np.random.randn(2, 3, 3, 3).astype(np.float32)
        A_mx = tk.array(A_np)
        d_mx = tk.linalg.det(A_mx, stream=tk.cpu)
        d_np = np.linalg.det(A_np)
        self.assertTrue(np.allclose(d_mx, d_np, atol=1e-4))

        # Integer input auto-promotes to float
        A = tk.array([[1, 2], [3, 4]])
        d = tk.linalg.det(A, stream=tk.cpu)
        self.assertTrue(np.allclose(d, -2.0))

        # float64
        A_np = np.random.randn(4, 4).astype(np.float64)
        A_mx = tk.array(A_np)
        d_mx = tk.linalg.det(A_mx, stream=tk.cpu)
        d_np = np.linalg.det(A_np)
        self.assertTrue(np.allclose(d_mx, d_np, atol=1e-10))

        # Singular 4x4 matrix (LU path): det should be 0
        A = tk.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
        d = tk.linalg.det(A, stream=tk.cpu)
        self.assertTrue(np.allclose(d, 0.0, atol=1e-5))

        # Singular 5x5 matrix (LU path)
        A_np = np.ones((5, 5), dtype=np.float32)
        A_mx = tk.array(A_np)
        d = tk.linalg.det(A_mx, stream=tk.cpu)
        self.assertTrue(np.allclose(d, 0.0, atol=1e-5))

        # Batched singular matrices (LU path)
        A_np = np.array([np.diag([1.0, 2.0, 0.0, 3.0]), np.eye(4, dtype=np.float32)])
        A_mx = tk.array(A_np)
        d_mx = tk.linalg.det(A_mx, stream=tk.cpu)
        d_np = np.linalg.det(A_np)
        self.assertTrue(np.allclose(d_mx, d_np, atol=1e-5))

        # Empty 0x0 matrix: det is the empty product = 1
        d = tk.linalg.det(tk.zeros((0, 0)), stream=tk.cpu)
        self.assertEqual(d.shape, ())
        self.assertEqual(float(d), 1.0)

        # Batched empty matrices: shape preserves batch dims
        d = tk.linalg.det(tk.zeros((3, 0, 0)), stream=tk.cpu)
        self.assertTrue(np.allclose(d, np.linalg.det(np.zeros((3, 0, 0)))))

        # Error: non-square
        with self.assertRaises(ValueError):
            tk.linalg.det(tk.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), stream=tk.cpu)

        # Error: 1D
        with self.assertRaises(ValueError):
            tk.linalg.det(tk.array([1.0, 2.0]), stream=tk.cpu)

        # Error: complex unsupported (small-matrix path)
        with self.assertRaises(ValueError):
            tk.linalg.det(tk.array([[1.0 + 1j, 2.0], [3.0, 4.0]]), stream=tk.cpu)

        # Error: complex unsupported (LU path)
        with self.assertRaises(ValueError):
            tk.linalg.det(tk.eye(4).astype(tk.complex64), stream=tk.cpu)

    def test_slogdet(self):
        # 2x2: det = -2 => sign = -1, logabsdet = log(2)
        A = tk.array([[1.0, 2.0], [3.0, 4.0]])
        sign, logabsdet = tk.linalg.slogdet(A, stream=tk.cpu)
        self.assertTrue(np.allclose(sign, -1.0))
        self.assertTrue(np.allclose(logabsdet, np.log(2.0), atol=1e-5))

        # Identity: sign = 1, logabsdet = 0
        A = tk.eye(4)
        sign, logabsdet = tk.linalg.slogdet(A, stream=tk.cpu)
        self.assertTrue(np.allclose(sign, 1.0))
        self.assertTrue(np.allclose(logabsdet, 0.0, atol=1e-6))

        # Compare with numpy for random matrices
        np.random.seed(42)
        for n in [1, 2, 3, 4, 5]:
            A_np = np.random.randn(n, n).astype(np.float32)
            A_mx = tk.array(A_np)
            sign_mx, logabs_mx = tk.linalg.slogdet(A_mx, stream=tk.cpu)
            sign_np, logabs_np = np.linalg.slogdet(A_np)
            with self.subTest(n=n):
                self.assertTrue(np.allclose(sign_mx, sign_np, atol=1e-5))
                self.assertTrue(np.allclose(logabs_mx, logabs_np, atol=1e-4))

        # Singular matrix 2x2 (fast path): sign = 0, logabsdet = -inf
        A = tk.array([[1.0, 2.0], [2.0, 4.0]])
        sign, logabsdet = tk.linalg.slogdet(A, stream=tk.cpu)
        self.assertEqual(float(sign), 0.0)
        self.assertEqual(float(logabsdet), float("-inf"))

        # Singular 4x4 matrix (LU path): sign = 0, logabsdet = -inf
        A = tk.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
        sign, logabsdet = tk.linalg.slogdet(A, stream=tk.cpu)
        self.assertEqual(float(sign), 0.0)
        self.assertEqual(float(logabsdet), float("-inf"))

        # Singular 5x5 matrix (LU path): all-ones matrix
        A = tk.array(np.ones((5, 5), dtype=np.float32))
        sign, logabsdet = tk.linalg.slogdet(A, stream=tk.cpu)
        self.assertEqual(float(sign), 0.0)
        self.assertEqual(float(logabsdet), float("-inf"))

        # Batched with mix of singular and non-singular (LU path)
        A_np = np.array([np.diag([1.0, 2.0, 0.0, 3.0]), np.eye(4, dtype=np.float32)])
        A_mx = tk.array(A_np)
        sign_mx, logabs_mx = tk.linalg.slogdet(A_mx, stream=tk.cpu)
        sign_np, logabs_np = np.linalg.slogdet(A_np)
        self.assertTrue(np.allclose(sign_mx, sign_np, atol=1e-5))
        # Check -inf for singular, 0.0 for identity
        self.assertEqual(float(logabs_mx[0]), float("-inf"))
        self.assertTrue(np.allclose(logabs_mx[1], 0.0, atol=1e-6))

        # Batched
        A_np = np.random.randn(3, 4, 4).astype(np.float32)
        A_mx = tk.array(A_np)
        sign_mx, logabs_mx = tk.linalg.slogdet(A_mx, stream=tk.cpu)
        sign_np, logabs_np = np.linalg.slogdet(A_np)
        self.assertTrue(np.allclose(sign_mx, sign_np, atol=1e-5))
        self.assertTrue(np.allclose(logabs_mx, logabs_np, atol=1e-4))

        # Multi-batch
        A_np = np.random.randn(2, 3, 3, 3).astype(np.float32)
        A_mx = tk.array(A_np)
        sign_mx, logabs_mx = tk.linalg.slogdet(A_mx, stream=tk.cpu)
        sign_np, logabs_np = np.linalg.slogdet(A_np)
        self.assertTrue(np.allclose(sign_mx, sign_np, atol=1e-5))
        self.assertTrue(np.allclose(logabs_mx, logabs_np, atol=1e-4))

        # Numerical stability: large matrix where det overflows
        # 0.1 * I_100 has det = 0.1^100 which underflows in float32
        # but slogdet should give sign=1, logabsdet = 100*log(0.1)
        n = 100
        A = tk.array(0.1) * tk.eye(n)
        sign, logabsdet = tk.linalg.slogdet(A, stream=tk.cpu)
        self.assertTrue(np.allclose(sign, 1.0))
        self.assertTrue(np.allclose(logabsdet, n * np.log(0.1), atol=1e-3))

        # Verify det = sign * exp(logabsdet) for non-singular cases
        A_np = np.random.randn(5, 5).astype(np.float32)
        A_mx = tk.array(A_np)
        sign_mx, logabs_mx = tk.linalg.slogdet(A_mx, stream=tk.cpu)
        det_mx = tk.linalg.det(A_mx, stream=tk.cpu)
        reconstructed = float(sign_mx) * np.exp(float(logabs_mx))
        self.assertTrue(np.allclose(float(det_mx), reconstructed, rtol=1e-4))

        # float64
        A_np = np.random.randn(4, 4).astype(np.float64)
        A_mx = tk.array(A_np)
        sign_mx, logabs_mx = tk.linalg.slogdet(A_mx, stream=tk.cpu)
        sign_np, logabs_np = np.linalg.slogdet(A_np)
        self.assertTrue(np.allclose(sign_mx, sign_np))
        self.assertTrue(np.allclose(logabs_mx, logabs_np, atol=1e-10))

        # Empty 0x0 matrix: sign = 1, logabsdet = 0 (empty product)
        sign, logabsdet = tk.linalg.slogdet(tk.zeros((0, 0)), stream=tk.cpu)
        self.assertEqual(sign.shape, ())
        self.assertEqual(logabsdet.shape, ())
        self.assertEqual(float(sign), 1.0)
        self.assertEqual(float(logabsdet), 0.0)

        # Batched empty matrices
        sign, logabsdet = tk.linalg.slogdet(tk.zeros((3, 0, 0)), stream=tk.cpu)
        sign_np, logabs_np = np.linalg.slogdet(np.zeros((3, 0, 0)))
        self.assertTrue(np.allclose(sign, sign_np))
        self.assertTrue(np.allclose(logabsdet, logabs_np))

        # Error: non-square
        with self.assertRaises(ValueError):
            tk.linalg.slogdet(
                tk.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), stream=tk.cpu
            )

        # Error: 1D
        with self.assertRaises(ValueError):
            tk.linalg.slogdet(tk.array([1.0, 2.0]), stream=tk.cpu)

        # Error: complex unsupported (small-matrix path)
        with self.assertRaises(ValueError):
            tk.linalg.slogdet(tk.array([[1.0 + 1j, 2.0], [3.0, 4.0]]), stream=tk.cpu)

        # Error: complex unsupported (LU path)
        with self.assertRaises(ValueError):
            tk.linalg.slogdet(tk.eye(4).astype(tk.complex64), stream=tk.cpu)


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
