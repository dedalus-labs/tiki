# Copyright © 2023-2024 Apple Inc.

import math
import unittest
from itertools import permutations

import numpy as np
import tiki as tk
import tiki_tests

# Ignore matmul warnings.
np.seterr(divide="ignore", over="ignore", invalid="ignore")


class TestBlas(tiki_tests.TIKITestCase):
    @property
    def dtypes(self):
        return ["float32", "float16"]

    def __gemm_test(
        self,
        shape_a,
        shape_b,
        np_dtype=np.float32,
        f_np_a=lambda x: x,
        f_np_b=lambda x: x,
        f_mx_a=lambda x: x,
        f_mx_b=lambda x: x,
    ):
        with self.subTest(
            dtype=np.dtype(np_dtype).name, shape_a=shape_a, shape_b=shape_b
        ):
            np.random.seed(42)
            scale = max(np.sum(shape_a), 128)
            a_np = np.random.normal(0.0, 1.0 / scale, shape_a).astype(np_dtype)
            b_np = np.random.normal(0.0, 1.0 / scale, shape_b).astype(np_dtype)

            a_mx = tk.array(a_np)
            b_mx = tk.array(b_np)

            a_np = f_np_a(a_np.astype(np.float32))
            b_np = f_np_b(b_np.astype(np.float32))
            a_mx = f_mx_a(a_mx)
            b_mx = f_mx_b(b_mx)

            out_npy = a_np @ b_np
            out_tiki = a_mx @ b_mx

            self.assertListEqual(list(out_npy.shape), list(out_tiki.shape))
            self.assertTrue(np.allclose(out_tiki, out_npy.astype(np_dtype), atol=1e-5))

    def test_matmul_unaligned(self):
        if not tk.is_available(tk.gpu):
            return

        for dtype in self.dtypes:
            np_dtype = getattr(np, dtype)
            base_shapes = [4, 8, 16, 32, 64, 128]
            perturbations = [-2, -1, 0, 1, 2]

            for dim in base_shapes:
                for p in perturbations:
                    shape_a = (dim + p, dim + p)
                    shape_b = (dim + p, dim + p)
                    self.__gemm_test(shape_a, shape_b, np_dtype)

    def test_matvec_unaligned(self):
        a = tk.random.normal(shape=(4, 128))
        b = tk.random.normal(shape=(129,))[1:]
        out = a @ b
        np_out = np.array(a) @ np.array(b)
        self.assertTrue(np.allclose(out, np_out))

    def test_matmul_shapes(self):
        if not tk.is_available(tk.gpu):
            return

        shapes = [
            (1, 2, 1, 1),
            (1, 1, 2, 1),
            (3, 23, 457, 3),
        ]

        if tk.default_device() == tk.gpu:
            shapes += [
                (16, 768, 768, 128),
                (1, 64, 64, 4096),
            ]

        for dtype in self.dtypes:
            np_dtype = getattr(np, dtype)

            for B, M, N, K in shapes:
                with self.subTest(transpose="nn"):
                    shape_a = (B, M, K)
                    shape_b = (B, K, N)
                    self.__gemm_test(shape_a, shape_b, np_dtype)

                with self.subTest(transpose="nt"):
                    shape_a = (B, M, K)
                    shape_b = (B, N, K)
                    self.__gemm_test(
                        shape_a,
                        shape_b,
                        np_dtype,
                        f_np_b=lambda x: np.transpose(x, (0, 2, 1)),
                        f_mx_b=lambda x: tk.transpose(x, (0, 2, 1)),
                    )

                with self.subTest(transpose="tn"):
                    shape_a = (B, K, M)
                    shape_b = (B, K, N)
                    self.__gemm_test(
                        shape_a,
                        shape_b,
                        np_dtype,
                        f_np_a=lambda x: np.transpose(x, (0, 2, 1)),
                        f_mx_a=lambda x: tk.transpose(x, (0, 2, 1)),
                    )

                with self.subTest(transpose="tt"):
                    shape_a = (B, K, M)
                    shape_b = (B, N, K)
                    self.__gemm_test(
                        shape_a,
                        shape_b,
                        np_dtype,
                        f_np_a=lambda x: np.transpose(x, (0, 2, 1)),
                        f_mx_a=lambda x: tk.transpose(x, (0, 2, 1)),
                        f_np_b=lambda x: np.transpose(x, (0, 2, 1)),
                        f_mx_b=lambda x: tk.transpose(x, (0, 2, 1)),
                    )

    def test_matmul(self):
        # Note: so far, matmul only works with floating-point types
        a = tk.array([[1.0, 2.0], [3.0, 4.0]])

        b = tk.array([[0.0, -1.0], [-3.0, 3.0]])

        expected = [[-6.0, 5.0], [-12.0, 9.0]]

        self.assertEqual((a @ b).tolist(), expected)
        self.assertEqual(tk.matmul(a, b).tolist(), expected)

        # Transposed matmul
        np.random.seed(0)
        a_npy = np.random.normal(0.0, 1.0 / 128, (128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (128, 16)).astype(np.float32)
        c_npy = a_npy @ np.transpose(b_npy, (1, 0))
        d_npy = np.transpose(a_npy, (1, 0)) @ b_npy

        a_tiki = tk.array(a_npy)
        b_tiki = tk.array(b_npy)
        c_tiki = a_tiki @ tk.transpose(b_tiki, (1, 0))
        d_tiki = tk.transpose(a_tiki, (1, 0)) @ b_tiki

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertListEqual(list(d_npy.shape), list(d_tiki.shape))

        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-6))
        self.assertTrue(np.allclose(d_tiki, d_npy, atol=1e-6))

    def test_matmul_dtypes(self):
        for dt in self.dtypes:
            a_npy = np.random.normal(0.0, 1.0 / 256, (16, 16, 16)).astype(
                getattr(np, dt)
            )
            b_npy = np.random.normal(0.0, 1.0 / 256, (16, 16, 16)).astype(
                getattr(np, dt)
            )
            a_tiki = tk.array(a_npy)
            b_tiki = tk.array(b_npy)

            c_npy = np.matmul(a_npy, b_npy, dtype=getattr(np, dt))
            c_tiki = a_tiki @ b_tiki

            self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-6))

    def test_matmul_batched(self):
        np.random.seed(0)
        # Batched matmul
        a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (32, 16, 16)).astype(np.float32)
        c_npy = a_npy @ b_npy

        a_tiki = tk.array(a_npy)
        b_tiki = tk.array(b_npy)
        c_tiki = a_tiki @ b_tiki

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-6))

        # Batched and transposed matmul
        b_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        c_npy = a_npy @ np.transpose(b_npy, (0, 2, 1))

        b_tiki = tk.array(b_npy)
        c_tiki = a_tiki @ tk.transpose(b_tiki, (0, 2, 1))

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-6))

        # Batched matmul with simple broadcast
        a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (16, 16)).astype(np.float32)
        c_npy = a_npy @ b_npy

        a_tiki = tk.array(a_npy)
        b_tiki = tk.array(b_npy)
        c_tiki = a_tiki @ b_tiki

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-6))

        # Both operands broadcasted
        d_npy = np.broadcast_to(b_npy, (5, 16, 16))
        d_tiki = tk.broadcast_to(b_tiki, (5, 16, 16))

        e_npy = d_npy @ d_npy
        e_tiki = d_tiki @ d_tiki

        self.assertListEqual(list(e_npy.shape), list(e_tiki.shape))
        self.assertTrue(np.allclose(e_tiki, e_npy, atol=1e-6))

        # Batched and transposed matmul with simple broadcast
        a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (128, 16)).astype(np.float32)
        a_tiki = tk.array(a_npy)
        b_tiki = tk.array(b_npy)

        c_npy = a_npy @ np.transpose(b_npy, (1, 0))
        c_tiki = a_tiki @ tk.transpose(b_tiki, (1, 0))

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-6))

        # Matmul with vector
        a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (16,)).astype(np.float32)
        a_tiki = tk.array(a_npy)
        b_tiki = tk.array(b_npy)

        c_npy = a_npy @ b_npy
        c_tiki = a_tiki @ b_tiki

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-6))

        # Test Multiheaded attention style matmul
        a_npy = np.random.normal(0.0, 1.0 / 128, (64, 16, 4, 32)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (64, 16, 4, 32)).astype(np.float32)
        a_tiki = tk.array(a_npy)
        b_tiki = tk.array(b_npy)

        a_npy = np.transpose(a_npy, (0, 2, 1, 3))
        b_npy = np.transpose(b_npy, (0, 2, 1, 3))
        a_tiki = tk.transpose(a_tiki, (0, 2, 1, 3))
        b_tiki = tk.transpose(b_tiki, (0, 2, 1, 3))

        c_npy = a_npy @ np.transpose(b_npy, (0, 1, 3, 2))
        c_tiki = a_tiki @ tk.transpose(b_tiki, (0, 1, 3, 2))
        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-6))

    def __gemv_test(
        self,
        shape_mat,
        shape_vec,
        np_dtype=np.float32,
        mat_first=True,
        np_mat_f=lambda x: x,
        np_vec_f=lambda x: x,
        tiki_mat_f=lambda x: x,
        tiki_vec_f=lambda x: x,
    ):
        with self.subTest(
            shape_mat=shape_mat, shape_vec=shape_vec, mat_first=mat_first
        ):
            np.random.seed(42)
            scale = max(np.sum(shape_mat), 32)
            mat_npy = np.random.normal(0.0, 1.0 / scale, shape_mat).astype(np_dtype)
            vec_npy = np.random.normal(0.0, 1.0 / scale, shape_vec).astype(np_dtype)

            mat_tiki = tk.array(mat_npy)
            vec_tiki = tk.array(vec_npy)

            mat_npy = np_mat_f(mat_npy)
            vec_npy = np_vec_f(vec_npy)
            mat_tiki = tiki_mat_f(mat_tiki)
            vec_tiki = tiki_vec_f(vec_tiki)

            if mat_first:
                out_npy = mat_npy @ vec_npy
                out_tiki = mat_tiki @ vec_tiki
            else:
                out_npy = vec_npy @ mat_npy
                out_tiki = vec_tiki @ mat_tiki

            # Due to some bug, numpy sometimes has NaNs on macOS
            # See https://github.com/ml-explore/mlx/pull/3063
            nans = np.isnan(out_npy)
            if np.any(nans):
                nan_ids = np.where(nans)
                tiki_nan_ids = tuple(tk.array(n) for n in nan_ids)
                out_npy[nan_ids] = 0.0
                out_tiki[tiki_nan_ids] = 0.0

            self.assertListEqual(list(out_npy.shape), list(out_tiki.shape))
            self.assertTrue(np.allclose(out_tiki, out_npy, atol=1e-5))

    def test_matrix_vector(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                np_dtype = getattr(np, dtype)

                # Basic square matrix test
                self.__gemv_test(
                    shape_mat=(64, 64), shape_vec=(64, 1), np_dtype=np_dtype
                )
                self.__gemv_test(
                    shape_mat=(64, 64),
                    shape_vec=(64, 1),
                    np_dtype=np_dtype,
                    mat_first=False,
                    np_vec_f=lambda x: np.transpose(x, (1, 0)),
                    tiki_vec_f=lambda x: tk.transpose(x, (1, 0)),
                )

                # Vector matrix product with aligned and unaligned shapes
                for in_len_base, out_len_base in (
                    (2, 2),
                    (32, 32),
                    (64, 64),
                    (2048, 2048),
                ):
                    for mi in (-1, 0, 1):
                        for mj in (-1, 0, 1):
                            # Vec mat
                            shape_mat = (in_len_base + mi, out_len_base + mj)
                            shape_vec = (1, in_len_base + mi)
                            self.__gemv_test(
                                shape_mat, shape_vec, mat_first=False, np_dtype=np_dtype
                            )

                            # Mat vec
                            shape_mat = (out_len_base + mj, in_len_base + mi)
                            shape_vec = (in_len_base + mi, 1)
                            self.__gemv_test(
                                shape_mat, shape_vec, mat_first=True, np_dtype=np_dtype
                            )

    def test_matrix_vector_batched(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                np_dtype = getattr(np, dtype)

                # Batched mat vec
                for shape_mat, shape_vec in (
                    ((32, 128, 64), (32, 64, 1)),
                    ((128, 64), (32, 64, 1)),
                    ((32, 128, 64), (64, 1)),
                    ((2, 1, 8, 1, 6, 128), (2, 1, 8, 4, 128, 1)),
                ):
                    self.__gemv_test(
                        shape_mat, shape_vec, mat_first=True, np_dtype=np_dtype
                    )

                # Batched vec mat
                for shape_vec, shape_mat in (
                    ((32, 1, 128), (32, 128, 64)),
                    ((32, 1, 128), (128, 64)),
                    ((1, 128), (32, 128, 64)),
                    ((1, 8, 4, 1, 128), (1, 8, 1, 128, 6)),
                ):
                    self.__gemv_test(
                        shape_mat, shape_vec, mat_first=False, np_dtype=np_dtype
                    )

    def test_matrix_vector_broadcast(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                np_dtype = getattr(np, dtype)

                # Different broadcasts mat vec
                for shape_mat, shape_vec in (
                    ((32, 64, 64), (32, 64, 1)),
                    ((64, 64), (32, 64, 1)),
                    ((32, 64, 64), (64, 1)),
                ):
                    self.__gemv_test(
                        shape_mat=(64, 64),
                        shape_vec=(64, 1),
                        np_dtype=np_dtype,
                        np_mat_f=(lambda mat_npy: np.broadcast_to(mat_npy, shape_mat)),
                        np_vec_f=(lambda vec_npy: np.broadcast_to(vec_npy, shape_vec)),
                        tiki_mat_f=(
                            lambda mat_tiki: tk.broadcast_to(mat_tiki, shape_mat)
                        ),
                        tiki_vec_f=(
                            lambda vec_tiki: tk.broadcast_to(vec_tiki, shape_vec)
                        ),
                    )

                # Different broadcasts vec mat
                for shape_vec, shape_mat in (
                    ((32, 1, 64), (32, 64, 64)),
                    ((32, 1, 64), (64, 64)),
                    ((1, 64), (32, 64, 64)),
                ):
                    self.__gemv_test(
                        shape_mat=(64, 64),
                        shape_vec=(1, 64),
                        np_dtype=np_dtype,
                        mat_first=False,
                        np_mat_f=lambda mat_npy: np.broadcast_to(mat_npy, shape_mat),
                        np_vec_f=lambda vec_npy: np.broadcast_to(vec_npy, shape_vec),
                        tiki_mat_f=lambda mat_tiki: tk.broadcast_to(
                            mat_tiki, shape_mat
                        ),
                        tiki_vec_f=lambda vec_tiki: tk.broadcast_to(
                            vec_tiki, shape_vec
                        ),
                    )

    def test_matrix_vector_attn(self):
        # Multi-query style attention check
        for dtype in self.dtypes:
            # fmt: off
            for (B,  D, n_kv_heads, factor,  qsl,  ksl) in (
                (1, 16,          8,      4,    1,  256),
                (1, 16,          8,      4,   32,  256),
                (1, 16,          8,      4,  256,    1),
                (4, 16,          8,      4,    1,  256),
                (4, 16,          8,      4,  256,    1),
            ):
            # fmt: on
                with self.subTest(
                        B=B, # Batch size
                        D=D, # Dimension of mm
                        n_kv_heads=n_kv_heads, # key-value heads
                        factor=factor, # factor to get query heads
                        qsl=qsl, # Query sequence length
                        ksl=ksl, # Key sequence length
                        dtype=dtype # Data type
                    ):

                    np_dtype = getattr(np, dtype)

                    # Fix shapes for kqv
                    n_q_heads = n_kv_heads * factor
                    Dk = D * n_kv_heads
                    Dq = D * n_q_heads
                    scale = 1. / math.sqrt(Dk)

                    shape_queries = (B, qsl, Dq)
                    shape_keys = (B, ksl, Dk)
                    shape_values = (B, ksl, Dk)

                    # Prepare numpy arrays
                    q_np = np.random.uniform(-scale, scale, size=shape_queries).astype(np_dtype)
                    k_np = np.random.uniform(-scale, scale, size=shape_keys).astype(np_dtype)
                    v_np = np.random.uniform(-scale, scale, size=shape_values).astype(np_dtype)

                    # Rearrange to move heads up
                    q_np_reshape = q_np.reshape(B, qsl, n_kv_heads, factor, -1).transpose(0, 2, 3, 1, 4)
                    k_np_reshape = k_np.reshape(B, ksl, n_kv_heads, 1, -1).transpose(0, 2, 3, 4, 1)
                    v_np_reshape = v_np.reshape(B, ksl, n_kv_heads, 1, -1).transpose(0, 2, 3, 1, 4)

                    # Do attn style matmul
                    s_np = q_np_reshape @ k_np_reshape
                    o_np = s_np @ v_np_reshape
                    o_np = o_np.transpose(0, 3, 1, 2, 4).reshape(B, qsl, -1)

                    # Test tiki
                    q_mx = tk.array(q_np)
                    k_mx = tk.array(k_np)
                    v_mx = tk.array(v_np)

                    # Rearrange to move heads up
                    q_mx_reshape = q_mx.reshape(B, qsl, n_kv_heads, factor, -1).transpose(0, 2, 3, 1, 4)
                    k_mx_reshape = k_mx.reshape(B, ksl, n_kv_heads, 1, -1).transpose(0, 2, 3, 4, 1)
                    v_mx_reshape = v_mx.reshape(B, ksl, n_kv_heads, 1, -1).transpose(0, 2, 3, 1, 4)

                    # Do attn style matmul
                    s_mx = q_mx_reshape @ k_mx_reshape
                    o_mx = (s_mx @ v_mx_reshape)
                    o_mx = o_mx.transpose(0, 3, 1, 2, 4).reshape(B, qsl, -1)

                    # Check against np
                    self.assertListEqual(list(s_np.shape), list(s_mx.shape))
                    self.assertTrue(np.allclose(s_np, s_mx, atol=1e-4))

                    self.assertListEqual(list(o_np.shape), list(o_mx.shape))
                    self.assertTrue(np.allclose(o_np, o_mx, atol=1e-4))

    def test_matrix_vector_edgecases(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                np_dtype = getattr(np, dtype)

                for in_vec_len in np.arange(1, 5):
                    for out_vec_len in np.arange(1, 5):
                        for batch_size in np.arange(1, 5):
                            with self.subTest(
                                problem_shape=(batch_size, in_vec_len, out_vec_len)
                            ):
                                # Matrix vector
                                with self.subTest(transpose=False):
                                    a_npy = np.ones(
                                        (batch_size, out_vec_len, in_vec_len),
                                        dtype=np_dtype,
                                    )
                                    b_npy = np.ones(
                                        (batch_size, in_vec_len, 1), dtype=np_dtype
                                    )
                                    for i in range(batch_size):
                                        b_npy[i] *= i + 1.0

                                    a_tiki, b_tiki = map(tk.array, [a_npy, b_npy])
                                    c_npy = a_npy @ b_npy
                                    c_tiki = a_tiki @ b_tiki

                                    self.assertListEqual(
                                        list(c_npy.shape), list(c_tiki.shape)
                                    )
                                    self.assertTrue(np.array_equal(c_tiki, c_npy))

                                # Vector matrix
                                with self.subTest(transpose=True):
                                    a_npy = np.ones(
                                        (batch_size, out_vec_len, in_vec_len),
                                        dtype=np_dtype,
                                    )
                                    b_npy = np.ones(
                                        (batch_size, 1, out_vec_len), dtype=np_dtype
                                    )
                                    for i in range(batch_size):
                                        b_npy[i] *= i + 1.0

                                    a_tiki, b_tiki = map(tk.array, [a_npy, b_npy])
                                    c_npy = b_npy @ a_npy
                                    c_tiki = b_tiki @ a_tiki

                                    self.assertListEqual(
                                        list(c_npy.shape), list(c_tiki.shape)
                                    )
                                    self.assertTrue(np.array_equal(c_tiki, c_npy))

    def test_dot_product(self):
        if tk.default_device() == tk.cpu:
            self.skipTest("requires GPU")

        def run_test(dtype, size, offset, atol):
            with self.subTest(dtype=str(dtype), size=size, offset=offset):
                np.random.seed(42)
                scale = size**-0.5
                a_mx = tk.array(
                    np.random.normal(0.0, scale, size + offset).astype(np.float32)
                ).astype(dtype)[offset:]
                b_mx = tk.array(
                    np.random.normal(0.0, scale, size + offset).astype(np.float32)
                ).astype(dtype)[offset:]

                expected = np.inner(
                    np.array(a_mx.astype(tk.float32)),
                    np.array(b_mx.astype(tk.float32)),
                )
                actual = np.array(tk.inner(a_mx, b_mx).astype(tk.float32))
                self.assertTrue(np.allclose(actual, expected, atol=atol))

        for dtype, atol in (
            (tk.float32, 1e-5),
            (tk.float16, 2e-3),
            (tk.bfloat16, 2e-3),
        ):
            for size in (1023, 1024, 1025, 16385, 131072, 1000000):
                for offset in (0, 1):
                    run_test(dtype, size, offset, atol)

    def test_wide_matmul(self):
        if tk.default_device() == tk.cpu:
            self.skipTest("requires GPU")

        # Eligible a @ b.T products of a few rows take the wide gemv route on metal;
        # cover numerical correctness across the routing boundaries: row tails,
        # K not divisible by 4, offset and sliced views, and batching.
        # Inputs scale as K**-0.5 so outputs stay O(K**-0.5); atol scales
        # with them, above output rounding for every dtype but below a
        # dropped reduction block.
        def run_test(dtype, shape_a, shape_b, f_np_b, f_mx_b):
            with self.subTest(dtype=str(dtype), shape_a=shape_a, shape_b=shape_b):
                np.random.seed(7)
                scale = shape_a[-1] ** -0.5
                a_mx = tk.array(
                    np.random.normal(0.0, scale, shape_a).astype(np.float32)
                ).astype(dtype)
                b_mx = tk.array(
                    np.random.normal(0.0, scale, shape_b).astype(np.float32)
                ).astype(dtype)
                a_np = np.array(a_mx.astype(tk.float32))
                b_np = np.array(b_mx.astype(tk.float32))

                out_np = a_np @ f_np_b(b_np)
                out_mx = (a_mx @ f_mx_b(b_mx)).astype(tk.float32)

                self.assertListEqual(list(out_np.shape), list(out_mx.shape))
                self.assertTrue(np.allclose(out_mx, out_np, atol=0.05 * scale))

        nt_np = lambda b: b.swapaxes(-1, -2)
        nt_mx = lambda b: tk.swapaxes(b, -1, -2)

        for dtype in (tk.float32, tk.float16, tk.bfloat16):
            for M in (1, 2, 3, 5, 8, 11, 16):
                for K, N in (
                    (64, 128),
                    (512, 128),
                    (2048, 256),
                    (2048, 32),
                    (2052, 1000),
                ):
                    run_test(dtype, (M, K), (N, K), nt_np, nt_mx)

            # K % 4 != 0 falls back to the general kernels
            run_test(dtype, (5, 514), (333, 514), nt_np, nt_mx)

            # sliced weights: leading dimension != K, plus offset views at 16-
            # and 8-byte alignment, and a slice with an odd vec4 tail
            run_test(
                dtype,
                (5, 512),
                (333, 576),
                lambda b: b[:, :512].swapaxes(-1, -2),
                lambda b: tk.swapaxes(b[:, :512], -1, -2),
            )
            run_test(
                dtype,
                (5, 512),
                (333, 512),
                lambda b: b[7:, :].swapaxes(-1, -2),
                lambda b: tk.swapaxes(b[7:, :], -1, -2),
            )
            run_test(
                dtype,
                (3, 512),
                (333, 520),
                lambda b: b[:, 4:516].swapaxes(-1, -2),
                lambda b: tk.swapaxes(b[:, 4:516], -1, -2),
            )
            run_test(
                dtype,
                (3, 2052),
                (129, 2056),
                lambda b: b[:, :2052].swapaxes(-1, -2),
                lambda b: tk.swapaxes(b[:, :2052], -1, -2),
            )

            # batched: regular, broadcast weights, and multi-dim batch
            run_test(dtype, (4, 3, 512), (4, 257, 512), nt_np, nt_mx)
            run_test(
                dtype,
                (4, 3, 512),
                (1, 257, 512),
                lambda b: np.broadcast_to(b, (4, 257, 512)).swapaxes(-1, -2),
                lambda b: tk.swapaxes(tk.broadcast_to(b, (4, 257, 512)), -1, -2),
            )
            run_test(dtype, (2, 3, 5, 512), (2, 3, 129, 512), nt_np, nt_mx)

    def test_mismatch_stride_mm(self):
        np.random.seed(0)
        a_npy = np.random.normal(0.0, 1.0 / 128, (4, 16, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (4, 16, 16)).astype(np.float32)

        a_tiki = tk.array(a_npy)
        b_tiki = tk.array(b_npy)

        # Matmul with batches
        c_npy = a_npy[::2, :, :] @ b_npy[1::2, :, :]
        c_tiki = a_tiki[::2, :, :] @ b_tiki[1::2, :, :]

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-5))

        # Matvec with batches
        c_npy = a_npy[::2, :, :] @ b_npy[1::2, :, 2:3]
        c_tiki = a_tiki[::2, :, :] @ b_tiki[1::2, :, 2:3]

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-5))

        # Matmul with slice
        c_npy = a_npy[:, :8, :] @ b_npy[:, :, :8]
        c_tiki = a_tiki[:, :8, :] @ b_tiki[:, :, :8]

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-5))

        # Matmul with slice
        c_npy = a_npy[:, :, :8] @ b_npy[:, :8, :]
        c_tiki = a_tiki[:, :, :8] @ b_tiki[:, :8, :]

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-5))

        # Matmul transpose with slice
        c_npy = a_npy[:, :8, :] @ b_npy[:, :8, :].swapaxes(-1, -2)
        c_tiki = a_tiki[:, :8, :] @ b_tiki[:, :8, :].swapaxes(-1, -2)

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-5))

        # Matmul transpose with slice
        c_npy = a_npy[:, :, :8] @ b_npy[:, :, :8].swapaxes(-1, -2)
        c_tiki = a_tiki[:, :, :8] @ b_tiki[:, :, :8].swapaxes(-1, -2)

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-5))

        # Matvec with slice
        c_npy = a_npy[:, :8, :] @ b_npy[:, :, 6:7]
        c_tiki = a_tiki[:, :8, :] @ b_tiki[:, :, 6:7]

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-5))

        # Matvec with slice
        c_npy = a_npy[:, :, :8] @ b_npy[:, 3:11, 2:3]
        c_tiki = a_tiki[:, :, :8] @ b_tiki[:, 3:11, 2:3]

        self.assertListEqual(list(c_npy.shape), list(c_tiki.shape))
        self.assertTrue(np.allclose(c_tiki, c_npy, atol=1e-5))

    def test_addmm(self):
        np.random.seed(0)
        # Batched matmul
        alpha = 0.5
        for beta in (1.0, 2.0):
            # c must broadcast to the output shape
            with self.assertRaises(ValueError):
                tk.addmm(tk.zeros((2, 2, 2)), tk.zeros((2, 2)), tk.zeros((2, 2)))

            # Regular batched case
            a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (32, 16, 16)).astype(np.float32)

            a_tiki = tk.array(a_npy)
            b_tiki = tk.array(b_npy)

            for c_shape in ((1,), (1, 16), (32, 1, 16), (1, 128, 16)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_tiki = tk.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_tiki = tk.addmm(c_tiki, a_tiki, b_tiki, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_tiki.shape))
                self.assertTrue(np.allclose(d_tiki, d_npy, atol=1e-5))

            # Batched and transposed matmul
            b_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
            b_tiki = tk.array(b_npy)

            for c_shape in ((1,), (32, 1, 128), (1, 128)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_tiki = tk.array(c_npy)

                b_np_t = np.transpose(b_npy, (0, 2, 1))
                b_mx_t = tk.transpose(b_tiki, (0, 2, 1))

                d_npy = alpha * (a_npy @ b_np_t) + beta * c_npy
                d_tiki = tk.addmm(c_tiki, a_tiki, b_mx_t, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_tiki.shape))
                self.assertTrue(np.allclose(d_tiki, d_npy, atol=1e-5))
            # Batched matmul with simple broadcast
            a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (16, 16)).astype(np.float32)

            a_tiki = tk.array(a_npy)
            b_tiki = tk.array(b_npy)

            for c_shape in ((1,), (1, 16), (32, 1, 16), (1, 128, 16)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_tiki = tk.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_tiki = tk.addmm(c_tiki, a_tiki, b_tiki, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_tiki.shape))
                self.assertTrue(np.allclose(d_tiki, d_npy, atol=1e-5))
            # Matmul with vector
            a_npy = np.random.normal(0.0, 1.0 / 128, (16,)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (32, 16, 128)).astype(np.float32)
            a_tiki = tk.array(a_npy)
            b_tiki = tk.array(b_npy)

            for c_shape in ((1,), (128,), (32, 128)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_tiki = tk.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_tiki = tk.addmm(c_tiki, a_tiki, b_tiki, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_tiki.shape))
                self.assertTrue(np.allclose(d_tiki, d_npy, atol=1e-5))

            # Matmul with vector
            a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (16,)).astype(np.float32)
            a_tiki = tk.array(a_npy)
            b_tiki = tk.array(b_npy)

            for c_shape in ((1,), (32, 128)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_tiki = tk.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_tiki = tk.addmm(c_tiki, a_tiki, b_tiki, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_tiki.shape))
                self.assertTrue(np.allclose(d_tiki, d_npy, atol=1e-5))

            # Split K specializtion
            a_npy = np.random.normal(0.0, 1.0 / 128, (64, 4096)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (4096, 32)).astype(np.float32)

            a_tiki = tk.array(a_npy)
            b_tiki = tk.array(b_npy)

            for c_shape in ((1,), (1, 32), (64, 1), (64, 32)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_tiki = tk.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_tiki = tk.addmm(c_tiki, a_tiki, b_tiki, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_tiki.shape))
                self.assertTrue(np.allclose(d_tiki, d_npy, atol=1e-5))

            # Transposed c
            a = tk.ones((10, 5)).T
            b = tk.ones((5, 5))
            out = tk.addmm(a, b, a, beta=beta, alpha=alpha)
            expected = beta * a + alpha * (b @ a)
            self.assertTrue(tk.allclose(expected, out))

            # Broadcast c
            a = tk.ones((5, 5))
            b = tk.ones((5, 5))
            c = tk.ones((1, 5))
            out = tk.addmm(c, a, b, beta=beta, alpha=alpha)
            expected = beta * c + alpha * (a @ b)
            self.assertTrue(tk.allclose(expected, out))

        # Test half precision
        for t, tol in [(tk.float16, 1e-3), (tk.bfloat16, 1e-2)]:
            c = tk.ones((32, 32)).astype(t)
            a = tk.random.uniform(shape=(32, 32)).astype(t)
            b = tk.random.uniform(shape=(32, 32)).astype(t)
            out = tk.addmm(c, a, b, alpha=0.5, beta=2.0)
            expected = 0.5 * (a @ b) + 2.0 * c
            self.assertTrue(tk.allclose(out, expected, rtol=tol, atol=tol))

    def test_wide_addmm(self):
        if tk.default_device() == tk.cpu:
            self.skipTest("requires GPU")

        # Eligible few-row addmm shapes take the wide gemv route on metal; cover
        # the axpby epilogue against bias shapes, scales, and batching.
        def run_test(dtype, B, M, K, N, c_shape, alpha, beta):
            with self.subTest(dtype=str(dtype), c_shape=c_shape, alpha=alpha):
                np.random.seed(3)
                shape_a = (M, K) if B is None else (B, M, K)
                shape_b = (N, K) if B is None else (B, N, K)
                scale = K**-0.5
                a_mx = tk.array(
                    np.random.normal(0.0, scale, shape_a).astype(np.float32)
                ).astype(dtype)
                b_mx = tk.array(
                    np.random.normal(0.0, scale, shape_b).astype(np.float32)
                ).astype(dtype)
                c_mx = tk.array(
                    np.random.normal(0.0, scale, c_shape).astype(np.float32)
                ).astype(dtype)
                a_np = np.array(a_mx.astype(tk.float32))
                b_np = np.array(b_mx.astype(tk.float32))
                c_np = np.array(c_mx.astype(tk.float32))

                out_np = alpha * (a_np @ b_np.swapaxes(-1, -2)) + beta * c_np
                out_mx = tk.addmm(
                    c_mx,
                    a_mx,
                    tk.swapaxes(b_mx, -1, -2),
                    alpha,
                    beta,
                ).astype(tk.float32)

                self.assertListEqual(list(out_np.shape), list(out_mx.shape))
                atol = 0.05 * (abs(alpha) + abs(beta)) * scale
                self.assertTrue(np.allclose(out_mx, out_np, atol=atol))

        for dtype in (tk.float32, tk.float16, tk.bfloat16):
            for M in (2, 5, 12):
                for c_shape in ((250,), (1, 250), (M, 250)):
                    for alpha, beta in ((1.0, 1.0), (2.5, 0.5), (1.0, 0.0)):
                        run_test(dtype, None, M, 512, 250, c_shape, alpha, beta)
            run_test(dtype, 3, 4, 512, 250, (3, 4, 250), 1.0, 1.0)

            # the epilogue must scale the accumulator before narrowing: a
            # product past the fp16 max rescued by alpha stays finite (M = 4
            # keeps the 2-byte shape routed on every supported generation)
            with self.subTest(dtype=str(dtype), case="alpha rescue"):
                a = tk.full((4, 512), 2.0, dtype=dtype)
                b = tk.full((250, 512), 100.0, dtype=dtype)
                c = tk.ones((4, 250), dtype=dtype)
                out = tk.addmm(c, a, tk.swapaxes(b, -1, -2), 0.125, 0.0)
                self.assertTrue(np.allclose(out.astype(tk.float32), 12800.0))

    def test_addmm_grad(self):
        def make_ref_addmm(alpha, beta):
            return lambda c, a, b: alpha * (a @ b) + beta * c

        def make_addmm(alpha, beta):
            return lambda c, a, b: tk.addmm(c, a, b, alpha, beta)

        # B, M, N, K
        shapes = ((1, 64, 32, 128), (4, 28, 24, 47), (1, 1, 24, 47))

        alpha = 2.0
        for beta in (1.0, 0.5):
            f_test = make_addmm(alpha, beta)
            f_ref = make_ref_addmm(alpha, beta)

            for B, M, N, K in shapes:
                cotan = tk.ones((B, M, N))
                c = tk.random.normal((B, M, N))
                a = tk.random.normal((B, M, K))
                b = tk.random.normal((B, K, N))

                out_ref, dout_ref = tk.vjp(
                    f_ref,
                    [c, a, b],
                    [cotan],
                )
                out_test, dout_test = tk.vjp(
                    f_test,
                    [c, a, b],
                    [cotan],
                )

                self.assertTrue(tk.allclose(out_ref[0], out_test[0], atol=1e-4).item())

                for r, t in zip(dout_ref, dout_test):
                    self.assertEqual(r.shape, t.shape)
                    self.assertTrue(tk.allclose(r, t, atol=1e-4).item())

    def test_empty_matmul(self):
        a = tk.array([[], []]).T
        b = tk.array([[1.0, 2.0], [2.0, 3.0]])
        c = a @ b
        tk.eval(c)
        self.assertEqual(c.shape, (0, 2))

        a = tk.array([[1.0, 2.0], [2.0, 3.0]])
        b = tk.array([[], []])
        c = a @ b
        tk.eval(c)
        self.assertEqual(c.shape, (2, 0))

        a = tk.array([[], []]).T
        b = tk.array([[], []])
        c = a @ b
        tk.eval(c)
        self.assertEqual(c.shape, (0, 0))

        c = tk.array(1.0, dtype=tk.float32)
        a = tk.array([], dtype=tk.float32)
        b = tk.array([], dtype=tk.float32)
        out = tk.addmm(c, a, b)
        self.assertEqual(out.item(), 1.0)
        self.assertEqual(out.shape, ())

        a = tk.ones((2, 0))
        b = tk.ones((0, 2))
        c = tk.ones((2, 2))

        test_cases = [
            (0.0, 1.0),
            (0.0, 2.0),
            (0.0, 0.5),
            (0.0, 0.0),
            (1.0, 2.0),
        ]

        for alpha, beta in test_cases:
            with self.subTest(alpha=alpha, beta=beta):
                result = tk.addmm(c, a, b, alpha=alpha, beta=beta)
                expected = c * beta  # a @ b = 0 for empty matrices
                self.assertTrue(tk.allclose(result, expected))

        shapes_tests = [
            ((3, 0), (0, 3), (3, 3)),
            ((5, 0), (0, 5), (5, 5)),
            ((1, 0), (0, 10), (1, 10)),
            ((10, 0), (0, 1), (10, 1)),
        ]

        for shape_a, shape_b, shape_c in shapes_tests:
            with self.subTest(shape_a=shape_a, shape_b=shape_b, shape_c=shape_c):
                a = tk.ones(shape_a)
                b = tk.ones(shape_b)
                c = tk.ones(shape_c)
                result = tk.addmm(c, a, b, alpha=0.5, beta=2.0)
                expected = c * 2.0
                self.assertTrue(tk.allclose(result, expected))

        a = tk.ones((2, 5, 0))
        b = tk.ones((2, 0, 5))
        c = tk.ones((2, 5, 5))
        result = tk.addmm(c, a, b, alpha=0.0, beta=3.0)
        expected = c * 3.0
        self.assertTrue(tk.allclose(result, expected))

    def test_block_masked_matmul(self):
        def ref_block_masked_mm(
            a, b, block_size, out_mask=None, lhs_mask=None, rhs_mask=None
        ):
            # Get mask adjusted shapes
            M = a.shape[-2]
            N = b.shape[-1]
            K = a.shape[-1]

            bsx_shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])

            # Expand mask dims
            def expand_mask(mask, block_size, Y, X):
                mask = tk.expand_dims(mask, (-3, -1))
                mask_shape = list(bsx_shape) + list(mask.shape[-4:])
                mask_shape[-1] = block_size
                x = mask_shape[-2] * block_size
                mask_shape[-3] = block_size
                y = mask_shape[-4] * block_size
                mask = tk.broadcast_to(mask, mask_shape)
                mask_shape = mask_shape[:-4] + [y, x]
                return mask.reshape(mask_shape)[..., :Y, :X]

            a_masked = a
            b_masked = b

            if lhs_mask is not None:
                lhs_mask = expand_mask(lhs_mask, block_size, M, K).astype(tk.float32)
                a_masked = lhs_mask * a_masked

            if rhs_mask is not None:
                rhs_mask = expand_mask(rhs_mask, block_size, K, N).astype(tk.float32)
                b_masked = rhs_mask * b_masked

            out = a_masked @ b_masked

            if out_mask is not None:
                out_mask = expand_mask(out_mask, block_size, M, N).astype(tk.float32)
                out = out * out_mask
            return out

        def run_test(a, b, block_size, out_mask, a_mask, b_mask, cotan):
            def f_ref(a_, b_):
                return ref_block_masked_mm(a_, b_, block_size, out_mask, a_mask, b_mask)

            def f_test(a_, b_):
                return tk.block_masked_mm(a_, b_, block_size, out_mask, a_mask, b_mask)

            out_ref, dout_ref = tk.vjp(f_ref, [a, b], [cotan])
            out_test, dout_test = tk.vjp(f_test, [a, b], [cotan])

            self.assertTrue(tk.allclose(out_ref[0], out_test[0], atol=1e-5).item())

            for r, t in zip(dout_ref, dout_test):
                self.assertEqual(r.shape, t.shape)
                self.assertTrue(tk.allclose(r, t, atol=1e-4).item())

        def run_test_mask_vjp(a, b, block_size, out_mask, a_mask, b_mask, cotan):
            def f_ref(a_, b_, a_mask_, b_mask_):
                return ref_block_masked_mm(
                    a_, b_, block_size, out_mask, a_mask_, b_mask_
                )

            def f_test(a_, b_, a_mask_, b_mask_):
                return tk.block_masked_mm(
                    a_, b_, block_size, out_mask, a_mask_, b_mask_
                )

            out_ref, dout_ref = tk.vjp(f_ref, [a, b, a_mask, b_mask], [cotan])
            out_test, dout_test = tk.vjp(f_test, [a, b, a_mask, b_mask], [cotan])

            tk.eval((out_ref, dout_ref, out_test, dout_test))

            self.assertTrue(tk.allclose(out_ref[0], out_test[0], atol=1e-5).item())

            for r, t in zip(dout_ref, dout_test):
                self.assertEqual(r.shape, t.shape)
                self.assertTrue(tk.allclose(r, t, atol=1e-4).item())

        def make_mask(tm_, tn_, batch, np_dtype):
            arr_np_mask = np.random.normal(size=batch + (tm_, tn_)).astype(np_dtype)
            arr_np_bool_mask = arr_np_mask < 0.0
            arr_np_mask[arr_np_bool_mask] = 0.0

            return tk.array(arr_np_bool_mask), tk.array(arr_np_mask)

        def test_shape(
            M,
            N,
            K,
            block_size,
            transpose=False,
            np_dtype=np.float32,
            batch_A=(),
            batch_B=(),
        ):
            with self.subTest(
                M=M,
                N=N,
                K=K,
                block_size=block_size,
                np_dtype=np_dtype,
                transpose=transpose,
                batch_A=batch_A,
                batch_B=batch_B,
            ):
                batch_out = np.broadcast_shapes(batch_A, batch_B)
                cotan = tk.ones(batch_out + (M, N))

                a_np = np.random.normal(size=batch_A + (M, K)).astype(np_dtype)
                b_np = np.random.normal(size=batch_B + (K, N)).astype(np_dtype)

                a_mx = tk.array(a_np)
                b_mx = tk.array(b_np)

                tm = (M + block_size - 1) // block_size
                tn = (N + block_size - 1) // block_size
                tk = (K + block_size - 1) // block_size

                a_mx_bool_mask, a_mx_mask = make_mask(tm, tk, batch_A, np_dtype)
                b_mx_bool_mask, b_mx_mask = make_mask(tk, tn, batch_B, np_dtype)
                out_mx_bool_mask, out_mx_mask = make_mask(tm, tn, batch_out, np_dtype)

                # Boolean block masks
                run_test(
                    a_mx,
                    b_mx,
                    block_size,
                    out_mx_bool_mask,
                    a_mx_bool_mask,
                    b_mx_bool_mask,
                    cotan,
                )
                run_test(a_mx, b_mx, block_size, out_mx_bool_mask, None, None, cotan)
                run_test(
                    a_mx, b_mx, block_size, None, a_mx_bool_mask, b_mx_bool_mask, cotan
                )

                # Float block masks
                run_test(
                    a_mx, b_mx, block_size, out_mx_mask, a_mx_mask, b_mx_mask, cotan
                )
                run_test(a_mx, b_mx, block_size, None, a_mx_mask, b_mx_mask, cotan)
                run_test_mask_vjp(
                    a_mx, b_mx, block_size, out_mx_mask, a_mx_mask, b_mx_mask, cotan
                )
                run_test_mask_vjp(
                    a_mx, b_mx, block_size, None, a_mx_mask, b_mx_mask, cotan
                )

        shapes = (
            (16, 16, 16, 32),
            (64, 64, 16, 32),
            (128, 128, 128, 32),
            (256, 256, 128, 64),
            (1, 128, 128, 32),
            (256, 1, 128, 64),
        )

        for M, N, K, block_size in shapes:
            test_shape(M, N, K, block_size)

        # Test broadcasting
        test_shape(64, 64, 64, 32, batch_A=(1, 2), batch_B=(2, 2))
        test_shape(1, 128, 128, 32, batch_A=(1, 2), batch_B=(2, 2))
        test_shape(128, 1, 128, 32, batch_A=(1, 2), batch_B=(2, 2))

        a_np = np.ones((128, 256)).astype(np.float32)
        b_np = np.ones((128, 1)).astype(np.float32)
        d_np = np.ones((1, 256)).astype(np.float32)
        a_mask_np = np.random.normal(size=(4, 8)).astype(np.float32)
        b_mask_np = np.ones((4, 1)).astype(np.bool_)
        d_mask_np = np.ones((1, 8)).astype(np.bool_)
        c_mask_np = np.random.normal(size=(8, 1)).astype(np.float32)
        e_mask_np = np.random.normal(size=(1, 4)).astype(np.float32)

        a_mask_np[a_mask_np < 0.0] = 0.0
        e_mask_np[e_mask_np < 0.0] = 0.0
        c_mask_np[c_mask_np < 0.0] = 0.0

        a_mx = tk.array(a_np)
        b_mx = tk.array(b_np)
        d_mx = tk.array(d_np)
        a_mask_mx = tk.array(a_mask_np)
        b_mask_mx = tk.array(b_mask_np)
        d_mask_mx = tk.array(d_mask_np)
        e_mask_mx = tk.array(e_mask_np)
        c_mask_mx = tk.array(c_mask_np)

        c_mx = tk.block_masked_mm(a_mx.T, b_mx, 32, c_mask_mx, a_mask_mx.T, b_mask_mx)
        e_mx = tk.block_masked_mm(d_mx, a_mx.T, 32, e_mask_mx, d_mask_mx, a_mask_mx.T)

        a_mask_np = np.broadcast_to(np.expand_dims(a_mask_np, (-3, -1)), (4, 32, 8, 32))
        a_mask_np = a_mask_np.reshape((128, 256))
        a_np *= a_mask_np

        c_np = a_np.T @ b_np
        e_np = d_np @ a_np.T

        c_mask_np = np.broadcast_to(np.expand_dims(c_mask_np, (-2)), (8, 32, 1))
        c_mask_np = c_mask_np.reshape((256, 1))
        c_np *= c_mask_np

        e_mask_np = np.broadcast_to(np.expand_dims(e_mask_np, (-1)), (1, 4, 32))
        e_mask_np = e_mask_np.reshape((1, 128))
        e_np *= e_mask_np

        self.assertTrue(np.allclose(c_mx, c_np, atol=1e-5))
        self.assertTrue(np.allclose(e_mx, e_np, atol=1e-5))

    def test_gather_matmul(self):
        def np_gather_mm(a, b, lhs_indices=None, rhs_indices=None):
            a = a.reshape((-1, a.shape[-2], a.shape[-1]))
            b = b.reshape((-1, b.shape[-2], b.shape[-1]))
            lhs_indices = lhs_indices or np.arange(a.shape[0])
            rhs_indices = rhs_indices or np.arange(b.shape[0])
            a = a[lhs_indices, :, :]
            b = b[rhs_indices, :, :]
            out = a @ b
            return out

        def test_shape(
            M,
            N,
            K,
            np_dtype=np.float32,
            batch_A=(),
            batch_B=(),
            lhs_indices=None,
            rhs_indices=None,
        ):
            with self.subTest(
                M=M,
                N=N,
                K=K,
                np_dtype=np_dtype,
                batch_A=batch_A,
                batch_B=batch_B,
                lhs_indices=lhs_indices,
                rhs_indices=rhs_indices,
            ):
                a_np = np.random.normal(size=batch_A + (M, K)).astype(np_dtype)
                b_np = np.random.normal(size=batch_B + (K, N)).astype(np_dtype)

                a_mx = tk.array(a_np)
                b_mx = tk.array(b_np)

                out_np = np_gather_mm(a_np, b_np, lhs_indices, rhs_indices)

                lhs_indices_mx = None if lhs_indices is None else tk.array(lhs_indices)
                rhs_indices_mx = None if rhs_indices is None else tk.array(rhs_indices)

                out_mx = tk.gather_mm(a_mx, b_mx, lhs_indices_mx, rhs_indices_mx)

                self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5))

        inputs = (
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (1,),
                "lhs_indices": None,
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (2,),
                "lhs_indices": None,
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (3,),
                "lhs_indices": (0, 2),
                "batch_B": (1,),
                "rhs_indices": (0,),
            },
            {
                "batch_A": (5,),
                "lhs_indices": (0, 2),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (4, 2),
                "lhs_indices": (
                    (7, 6),
                    (5, 4),
                    (1, 2),
                ),
                "batch_B": (4, 1),
                "rhs_indices": ((2,), (0,), (1,)),
            },
        )

        for kwargs in inputs:
            test_shape(32, 32, 32, **kwargs)
            test_shape(16, 1, 16, **kwargs)

        # Add tests for broadcasting
        a_np = np.random.normal(size=(5, 32, 32)).astype(np.float32)
        b_np = np.random.normal(size=(3, 32, 32)).astype(np.float32)
        a_mx = tk.array(a_np)
        b_mx = tk.array(b_np)

        # Numpy
        a_np = a_np.reshape((5, 1, 32, 32))
        b_np = b_np.reshape((1, 3, 32, 32))

        a_np = np.broadcast_to(a_np, (5, 4, 32, 32))
        b_np = np.broadcast_to(b_np, (2, 3, 32, 32)).swapaxes(1, 0)

        lhs_indices = [0, 13, 12]
        rhs_indices = [0, 3, 5]

        out_np = np_gather_mm(a_np, b_np, lhs_indices, rhs_indices)

        # Tiki
        a_mx = a_mx.reshape((5, 1, 32, 32))
        b_mx = b_mx.reshape((1, 3, 32, 32))

        a_mx = tk.broadcast_to(a_mx, (5, 4, 32, 32))
        b_mx = tk.broadcast_to(b_mx, (2, 3, 32, 32)).swapaxes(1, 0)

        lhs_indices_mx = tk.array(lhs_indices)
        rhs_indices_mx = tk.array(rhs_indices)

        out_mx = tk.gather_mm(a_mx, b_mx, lhs_indices_mx, rhs_indices_mx)

        self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5))

        # Gemv test
        a_np = np.random.normal(size=(5, 1, 32)).astype(np.float32)
        b_np = np.random.normal(size=(3, 16, 32)).astype(np.float32)
        a_mx = tk.array(a_np)
        b_mx = tk.array(b_np)

        lhs_indices = [3, 1]
        rhs_indices = [0, 2]

        b_np_t = np.swapaxes(b_np, -1, -2)
        out_np = np_gather_mm(a_np, b_np_t, lhs_indices, rhs_indices)

        lhs_indices_mx = tk.array(lhs_indices)
        rhs_indices_mx = tk.array(rhs_indices)

        b_mx_t = tk.swapaxes(b_mx, -1, -2)
        out_mx = tk.gather_mm(a_mx, b_mx_t, lhs_indices_mx, rhs_indices_mx)

        self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5))

    def test_gather_mm_blocks(self):
        if tk.default_device() == tk.cpu:
            self.skipTest("requires GPU")

        # Eligible gathered products with few-row blocks route to the wide
        # gather on metal; check block indexing against a per-entry reference.
        def run_test(dtype, G, M, K, N, E, idx_shape):
            with self.subTest(dtype=str(dtype), G=G, M=M, idx=idx_shape):
                np.random.seed(11)
                scale = K**-0.5
                a_mx = tk.array(
                    np.random.normal(0.0, scale, (G, M, K)).astype(np.float32)
                ).astype(dtype)
                w_mx = tk.array(
                    np.random.normal(0.0, scale, (E, N, K)).astype(np.float32)
                ).astype(dtype)
                a_np = np.array(a_mx.astype(tk.float32))
                w_np = np.array(w_mx.astype(tk.float32))
                rhs = np.random.randint(0, E, size=idx_shape).astype(np.uint32)

                out_np = np.stack(
                    [a_np[i % G] @ w_np[r].T for i, r in enumerate(rhs.reshape(-1))]
                ).reshape(*idx_shape, M, N)
                out_mx = tk.gather_mm(
                    a_mx.reshape(*idx_shape, M, K),
                    tk.swapaxes(w_mx, -1, -2),
                    None,
                    tk.array(rhs),
                ).astype(tk.float32)

                self.assertListEqual(list(out_np.shape), list(out_mx.shape))
                self.assertTrue(np.allclose(out_mx, out_np, atol=0.05 * scale))

        for dtype in (tk.float32, tk.float16, tk.bfloat16):
            for M in (2, 4, 5, 11):
                run_test(dtype, 6, M, 2048, 1024, 8, (6,))
            run_test(dtype, 6, 3, 2048, 1024, 8, (2, 3))

            # scalar indices: batch_ndim == 0 must still bind index strides
            with self.subTest(dtype=str(dtype), idx="scalar"):
                np.random.seed(11)
                scale = 512**-0.5
                a_mx = tk.array(
                    np.random.normal(0.0, scale, (4, 512)).astype(np.float32)
                ).astype(dtype)
                w_mx = tk.array(
                    np.random.normal(0.0, scale, (8, 129, 512)).astype(np.float32)
                ).astype(dtype)
                out_np = (
                    np.array(a_mx.astype(tk.float32))
                    @ np.array(w_mx[3].astype(tk.float32)).T
                )
                out_mx = tk.gather_mm(
                    a_mx,
                    tk.swapaxes(w_mx, -1, -2),
                    None,
                    tk.array(3, dtype=tk.uint32),
                ).astype(tk.float32)
                self.assertListEqual(list(out_np.shape), list(out_mx.shape))
                self.assertTrue(np.allclose(out_mx, out_np, atol=0.05 * scale))

    def test_gather_matmul_grad(self):
        lhs_indices = tk.array([[7, 6], [4, 1], [0, 2]], dtype=tk.uint32)
        rhs_indices = tk.array([[2], [0], [1]], dtype=tk.uint32)

        def f_ref(a, b):
            lhs_indices_ = tk.broadcast_to(lhs_indices, (3, 2))
            rhs_indices_ = tk.broadcast_to(rhs_indices, (3, 2))
            M = a.shape[-2]
            N = b.shape[-1]
            K = a.shape[-1]

            a = a.reshape((-1, M, K))
            b = b.reshape((-1, K, N))

            a = tk.take(a, lhs_indices_, 0)
            b = tk.take(b, rhs_indices_, 0)

            return a @ b

        def f_test(a, b):
            return tk.gather_mm(a, b, lhs_indices, rhs_indices)

        a_mx = tk.random.normal((4, 2, 32, 32))
        b_mx = tk.random.normal((4, 1, 32, 32))

        out_test = f_test(a_mx, b_mx)
        out_ref = f_ref(a_mx, b_mx)

        self.assertTrue(tk.allclose(out_test, out_ref, atol=1e-5))

        cotan = tk.ones_like(out_test)
        out_ref, dout_ref = tk.vjp(
            f_ref,
            [a_mx, b_mx],
            [cotan],
        )
        out_test, dout_test = tk.vjp(
            f_test,
            [a_mx, b_mx],
            [cotan],
        )

        for r, t in zip(dout_ref, dout_test):
            self.assertEqual(r.shape, t.shape)
            self.assertTrue(tk.allclose(r, t, atol=1e-4).item())

    def test_gather_matmul_index_vjp_requires_stop_gradient(self):
        a = tk.ones((4, 1, 2, 2))
        b = tk.ones((4, 1, 2, 2))

        def fun(w):
            indices = tk.reshape(tk.argsort(w)[:2], (1, 2))
            return tk.gather_mm(a, b, indices, indices).sum()

        with self.assertRaisesRegex(ValueError, "stop_gradient"):
            tk.grad(fun)(tk.array([3.0, 1.0, 2.0, 0.0]))

        def fun_stopped(w):
            indices = tk.stop_gradient(tk.reshape(tk.argsort(w)[:2], (1, 2)))
            return tk.gather_mm(a, b, indices, indices).sum()

        grad = tk.grad(fun_stopped)(tk.array([3.0, 1.0, 2.0, 0.0]))
        self.assertTrue(tk.array_equal(grad, tk.zeros((4,))))

    def test_gather_mm_sorted(self):
        def gather_mm_ref(a, b, rhs):
            b = b[rhs]
            return a @ b

        def gather_mm_test(a, b, rhs):
            return tk.gather_mm(a, b, rhs_indices=rhs, sorted_indices=True)

        dtypes = [(tk.float32, 1e-4)]
        if tk.cuda.is_available():
            dtypes += [
                (tk.float16, 1e-3),
                (tk.bfloat16, 1e-2),
            ]

        for b_transposed in (True, False):
            for dtype, tol in dtypes:
                with self.subTest(b_transposed=b_transposed, dtype=dtype):
                    a = tk.random.normal((100, 1, 100), dtype=dtype)
                    b = tk.random.normal((8, 100, 100), dtype=dtype)
                    if b_transposed:
                        b = b.swapaxes(-1, -2)
                    rhs = tk.sort(tk.random.randint(0, 8, shape=(100,)))

                    c1 = gather_mm_ref(a, b, rhs)
                    c2 = gather_mm_test(a, b, rhs)
                    self.assertTrue(tk.allclose(c1, c2, rtol=tol, atol=tol))

    def test_gather_mm_sorted_vjp(self):
        def gather_mm_ref(a, b, rhs):
            b = b[rhs]
            return a @ b

        def gather_mm_test(a, b, rhs):
            return tk.gather_mm(a, b, rhs_indices=rhs, sorted_indices=True)

        a = tk.random.normal((100, 1, 100))
        b = tk.random.normal((8, 100, 100))
        rhs = tk.sort(tk.random.randint(0, 8, shape=(100,)))

        cotan = tk.random.normal((100, 1, 100))
        c1, dc1 = tk.vjp(
            lambda a, b: gather_mm_ref(a, b, rhs),
            [a, b],
            [cotan],
        )
        c2, dc2 = tk.vjp(
            lambda a, b: gather_mm_test(a, b, rhs),
            [a, b],
            [cotan],
        )
        self.assertTrue(tk.allclose(c1[0], c2[0], atol=1e-4))
        self.assertTrue(tk.allclose(dc1[0], dc2[0], atol=1e-4))
        self.assertTrue(tk.allclose(dc1[1], dc2[1], atol=1e-4))

    def test_segmented_mm(self):
        def segmented_mm_ref(a, b, s):
            s = s.tolist()
            c = []
            for s1, s2 in s:
                c.append(a[:, s1:s2] @ b[s1:s2, :])
            return tk.stack(c, axis=0)

        shapes = [
            (10, 10, 10),
            (10, 10, 1000),
            (1000, 1000, 1000),
        ]
        all_segments = [[0, 0, 1.0], [0, 0.5, 1.0], [r / 9 for r in range(10)]]

        for M, N, K in shapes:
            for s in all_segments:
                segments = []
                for i in range(len(s) - 1):
                    segments.append([s[i], s[i + 1]])
                segments = tk.array(segments)
                segments = tk.minimum(K - 1, (K * segments).astype(tk.uint32))
                a = tk.random.normal((M, K))
                b = tk.random.normal((K, N))
                c1 = segmented_mm_ref(a, b, segments)
                c2 = tk.segmented_mm(a, b, segments)
                self.assertTrue(tk.allclose(c1, c2, atol=1e-4))

                a = tk.random.normal((K, M))
                b = tk.random.normal((K, N))
                c1 = segmented_mm_ref(a.T, b, segments)
                c2 = tk.segmented_mm(a.T, b, segments)
                self.assertTrue(tk.allclose(c1, c2, atol=1e-4))

                a = tk.random.normal((M, K))
                b = tk.random.normal((N, K))
                c1 = segmented_mm_ref(a, b.T, segments)
                c2 = tk.segmented_mm(a, b.T, segments)
                self.assertTrue(tk.allclose(c1, c2, atol=1e-4))

                a = tk.random.normal((K, M))
                b = tk.random.normal((N, K))
                c1 = segmented_mm_ref(a.T, b.T, segments)
                c2 = tk.segmented_mm(a.T, b.T, segments)
                self.assertTrue(tk.allclose(c1, c2, atol=1e-4))

        with self.assertRaises(ValueError):
            a = tk.ones((2, 10, 10))
            s = tk.array([[0, 5], [5, 10]]).astype(tk.uint32)
            tk.segmented_mm(a, a, s)

        a = tk.ones((10, 1000))
        s = tk.random.randint(0, 16, shape=(1000,))
        s = tk.zeros(16, dtype=s.dtype).at[s].add(1)
        s = tk.sort(s)
        s = tk.cumsum(s)
        s = tk.concatenate([tk.array([0]), s])
        s = tk.as_strided(s, (16, 2), (1, 1))
        s = tk.reshape(s, (2, 2, 4, 2))
        c = tk.segmented_mm(a, a.T, s)
        self.assertEqual(c.shape, (2, 2, 4, 10, 10))

    def test_gemv_gemm_same_precision(self):
        tk.random.seed(0)
        N = 256
        if tk.is_available(tk.gpu):
            t = tk.bfloat16
            a = tk.random.normal([1, N]).astype(t)
            b = tk.concatenate([a, a], axis=0).astype(t)
            c = tk.random.normal([N, 64]).astype(t)
            out_gemv = a @ c
            out_gemm = (b @ c)[0]
            self.assertTrue(tk.allclose(out_gemv, out_gemm))

    def test_complex_gemv(self):
        M = 16
        N = 50

        def rand(shape):
            return tk.random.uniform(shape=shape) + 1j * tk.random.uniform(shape=shape)

        a = rand((M, N))
        b = rand((N, 1))
        c = tk.matmul(a, b)
        c_np = np.matmul(a, b)
        self.assertTrue(np.allclose(c, c_np))

        # Transposed
        a = rand((N, M))
        b = rand((N, 1))
        c = tk.matmul(a.T, b)
        c_np = np.matmul(np.array(a).T, b)
        self.assertTrue(np.allclose(c, c_np))

        # Check shapes
        a = tk.random.normal((2, 3)).astype(tk.complex64)
        b = tk.random.normal((3,))
        self.assertEqual((a @ b).shape, (2,))

        a = tk.random.normal((2, 3)).astype(tk.complex64)
        b = tk.random.normal((3,))
        c = tk.random.normal((2,))
        self.assertEqual(tk.addmm(c, a, b).shape, (2,))

    def test_complex_gemm(self):
        M = 16
        K = 50
        N = 32

        def rand(shape):
            return tk.random.uniform(shape=shape) + 1j * tk.random.uniform(shape=shape)

        a = rand((M, K))
        b = rand((K, N))
        c = tk.matmul(a, b)
        c_np = np.matmul(a, b)
        self.assertTrue(np.allclose(c, c_np))

        # Test addmm
        a = rand((M, K))
        b = rand((K, N))
        c = rand((M, N))
        out = tk.addmm(c, a, b, 2.0, 2.0)
        out_np = 2.0 * np.matmul(a, b) + 2.0 * c
        self.assertTrue(np.allclose(out, out_np))

        # complex with real
        a = rand((M, K)).real
        b = rand((K, N))
        c = tk.matmul(a, b)
        c_np = np.matmul(a, b)
        self.assertTrue(np.allclose(out, out_np))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
