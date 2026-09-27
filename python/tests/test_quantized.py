# Copyright © 2023-2026 Apple Inc.

import math
import os
import platform
import subprocess
import unittest
from itertools import product

import tiki as tk
import tiki_tests


def is_m1_mac():
    if platform.system() != "Darwin":
        return False
    cmd = "sysctl -n machdep.cpu.brand_string"
    cpu = subprocess.check_output(cmd, shell=True).decode().strip()
    return cpu.startswith("Apple M1")


class TestQuantized(tiki_tests.TIKITestCase):
    def test_quantize_dequantize(self):
        w = tk.random.normal(shape=(128, 512))
        for gs in [32, 64, 128]:
            for b in [2, 3, 5, 6, 4, 8]:
                with self.subTest(gs=gs, b=b):
                    w_q, scales, biases = tk.quantize(w, group_size=gs, bits=b)
                    w_hat = tk.dequantize(w_q, scales, biases, gs, b)
                    errors = (w - w_hat).abs().reshape(*scales.shape, -1)
                    eps = 1e-6
                    self.assertTrue((errors <= (scales[..., None] + eps).abs()).all())

        # test quantize/dequantize 0s
        a = tk.zeros((256, 512))
        for gs in [32, 64, 128]:
            for b in [2, 3, 4, 5, 6, 8]:
                w_q, scales, biases = tk.quantize(a, gs, b)
                a_hat = tk.dequantize(w_q, scales, biases, gs, b)
                self.assertTrue(tk.all(a_hat == 0))

        # slices
        if tk.default_device() == tk.gpu:
            w = tk.random.normal(shape=(2, 256, 32))
            quant = {"group_size": 32, "bits": 4}
            wq, scales, biases = tk.quantize(w, **quant)
            wq_s = wq[:, :16, :]
            scales_s = scales[:, :16, :]
            biases_s = biases[:, :16, :]
            dq_cpu = tk.dequantize(wq_s, scales_s, biases_s, **quant, stream=tk.cpu)
            dq_gpu = tk.dequantize(wq_s, scales_s, biases_s, **quant, stream=tk.gpu)
            self.assertTrue(tk.abs(dq_cpu - dq_gpu).max().item() < 1e-6)

    def test_mxfp4_quantize_dequantize(self):
        lut = tk.array(
            [
                +0.0,
                +0.5,
                +1.0,
                +1.5,
                +2.0,
                +3.0,
                +4.0,
                +6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        w = lut[tk.random.randint(0, 16, shape=(128, 512))]
        w = w.reshape(-1, 32)
        w[:, 0] = 6
        w = (w + 3e-6).astype(tk.bfloat16)

        # Invalid bits / group size
        with self.assertRaises(ValueError):
            tk.quantize(w, bits=3, mode="mxfp4")

        with self.assertRaises(ValueError):
            tk.quantize(w, group_size=64, mode="mxfp4")

        w_q, scales = tk.quantize(w, mode="mxfp4")
        with self.assertRaises(ValueError):
            tk.dequantize(w_q, scales, bits=3, mode="mxfp4")

        with self.assertRaises(ValueError):
            tk.dequantize(w_q, scales, group_size=64, mode="mxfp4")

        # Invalid output type
        with self.assertRaises(ValueError):
            tk.dequantize(
                w_q, scales, group_size=32, bits=4, mode="mxfp4", dtype=tk.int32
            )

        w_hat = tk.dequantize(w_q, scales, mode="mxfp4")
        self.assertTrue(tk.allclose(w, w_hat, rtol=1e-5, atol=1e-5))

        # test quantize/dequantize 0s
        a = tk.zeros((256, 512))
        w_q, scales = tk.quantize(a, mode="mxfp4")
        w_hat = tk.dequantize(w_q, scales, mode="mxfp4")
        self.assertTrue(tk.all(w_hat == 0))

    def test_mxfp8_quantize_dequantize(self):
        w = 2 * tk.random.uniform(shape=(512, 32)) - 1
        w = w.astype(tk.bfloat16)

        # Invalid bits / group size
        with self.assertRaises(ValueError):
            tk.quantize(w, bits=3, mode="mxfp8")

        with self.assertRaises(ValueError):
            tk.quantize(w, group_size=32, bits=7, mode="mxfp8")
        w_q, scales = tk.quantize(w, group_size=32, mode="mxfp8")

        with self.assertRaises(ValueError):
            tk.dequantize(w_q, scales, group_size=16, mode="mxfp8")

        with self.assertRaises(ValueError):
            tk.dequantize(w_q, scales, bits=4, mode="mxfp8")

        w_hat = tk.dequantize(w_q, scales, mode="mxfp8")

        self.assertTrue(tk.allclose(w, w_hat, rtol=1e-1, atol=1e-1))

        # test quantize/dequantize 0s
        a = tk.zeros((256, 512))
        w_q, scales = tk.quantize(a, mode="mxfp8")
        w_hat = tk.dequantize(w_q, scales, mode="mxfp8")
        self.assertTrue(tk.all(w_hat == 0))

    def test_mxfp8_block_scale_does_not_saturate(self):
        # E4M3 has three mantissa bits, so an in-range element loses at most
        # half a step, 6.25%. More than that means the block scale rounded
        # below amax/448 and the block maximum saturated.
        tk.random.seed(0)
        group_size = 32
        n_blocks = 512

        # Sweep the block magnitude across one binade so both scale rounding
        # directions are covered.
        w = tk.random.normal(shape=(n_blocks, group_size))
        w = w * tk.exp(tk.arange(n_blocks) / n_blocks * math.log(2.0)).reshape(-1, 1)

        w_q, scales = tk.quantize(w, group_size=group_size, mode="mxfp8")
        w_hat = tk.dequantize(w_q, scales, group_size=group_size, mode="mxfp8")

        # Quantization is monotone in |w|, so a block's largest output is the
        # reconstruction of its largest input.
        amax = tk.max(tk.abs(w), axis=1)
        rel = tk.abs(amax - tk.max(tk.abs(w_hat), axis=1)) / amax
        self.assertLess(tk.max(rel).item(), 0.0626)

    def test_nvfp4_quantize_dequantize(self):
        lut = tk.array(
            [
                +0.0,
                +0.5,
                +1.0,
                +1.5,
                +2.0,
                +3.0,
                +4.0,
                +6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        w = lut[tk.random.randint(0, 16, shape=(128, 512))]
        w = w.reshape(-1, 16)
        w[:, 0] = 6
        w = (w + 3e-6).astype(tk.bfloat16)

        # Invalid bits / group size
        with self.assertRaises(ValueError):
            tk.quantize(w, bits=3, mode="nvfp4")

        with self.assertRaises(ValueError):
            tk.quantize(w, group_size=64, mode="nvfp4")

        w_q, scales = tk.quantize(w, mode="nvfp4")

        with self.assertRaises(ValueError):
            tk.dequantize(w_q, scales, bits=3, mode="nvfp4")

        with self.assertRaises(ValueError):
            tk.dequantize(w_q, scales, group_size=32, mode="nvfp4")

        w_hat = tk.dequantize(w_q, scales, mode="nvfp4")
        self.assertTrue(tk.allclose(w, w_hat, rtol=1e-5, atol=1e-5))

        # A scale shared across a 32-value SIMD group instead of computed
        # per 16-value group cannot represent the low-magnitude groups.
        alternating = tk.zeros((64, 16), dtype=tk.bfloat16)
        alternating[::2] = 6 * 2**-9
        alternating[1::2] = 6.0
        w_q, scales = tk.quantize(alternating, mode="nvfp4")
        w_hat = tk.dequantize(w_q, scales, mode="nvfp4", dtype=tk.bfloat16)
        self.assertTrue(tk.allclose(alternating, w_hat, rtol=1e-5, atol=1e-6))

        # test quantize/dequantize 0s
        a = tk.zeros((256, 512))
        w_q, scales = tk.quantize(a, mode="nvfp4")
        w_hat = tk.dequantize(w_q, scales, mode="nvfp4")
        self.assertTrue(tk.all(w_hat == 0))

        # Test nvfp4 quantize/dequantize with tensor-scale global_scale
        global_scale = w.abs().max().astype(tk.float32)

        w_q, scales = tk.quantize(w, mode="nvfp4", global_scale=global_scale)
        w_hat = tk.dequantize(
            w_q, scales, group_size=16, bits=4, mode="nvfp4", global_scale=global_scale
        )
        self.assertTrue(tk.allclose(w, w_hat, rtol=1e-5, atol=1e-5))

    def test_qqmv(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        tests = product(
            [256, 512, 67],  # M
            [64, 256, 512],  # N
            ["nvfp4", "mxfp8"],  # mode
        )
        for M, N, mode in tests:
            with self.subTest(shape=(M, N), mode=mode):
                x_shape = (1, N)
                w_shape = (M, N)

                # TODO: Fix qmv with global scale in CPU backend.
                has_global_scale = mode == "nvfp4" and tk.default_device() == tk.gpu

                x = tk.random.normal(shape=x_shape, key=k1)
                global_scale_x = tk.max(tk.abs(x)) if has_global_scale else None
                x_hat = tk.dequantize(
                    *tk.quantize(x, mode=mode, global_scale=global_scale_x),
                    mode=mode,
                    dtype=tk.float32,
                    global_scale=global_scale_x,
                )

                w = tk.random.normal(shape=w_shape, key=k2)
                global_scale_w = tk.max(tk.abs(w)) if has_global_scale else None
                w_q, scales = tk.quantize(w, mode=mode, global_scale=global_scale_w)
                w_hat = tk.dequantize(
                    w_q,
                    scales,
                    mode=mode,
                    global_scale=global_scale_w,
                    dtype=tk.float32,
                )
                y_q = tk.qqmm(
                    x,
                    w_q,
                    scales,
                    mode=mode,
                    global_scale_x=global_scale_x,
                    global_scale_w=global_scale_w,
                )
                y_hat = x_hat @ tk.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qqmm(self):
        if tk.default_device() == tk.cpu:
            self.skipTest("Not implemented for CPU")
            return

        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        tests = product(
            [8, 32, 33, 64],  # M
            [128, 256],  # N
            [128, 256],  # K
            ["nvfp4", "mxfp8"],  # mode
        )
        for M, N, K, mode in tests:
            with self.subTest(shape=(M, N, K), mode=mode):
                x_shape = (M, K)
                w_shape = (N, K)

                x = tk.random.normal(shape=x_shape, key=k1)
                global_scale_x = tk.max(tk.abs(x)) if mode == "nvfp4" else None
                x_hat = tk.dequantize(
                    *tk.quantize(x, mode=mode, global_scale=global_scale_x),
                    mode=mode,
                    dtype=tk.float32,
                    global_scale=global_scale_x,
                )

                w = tk.random.normal(shape=w_shape, key=k2)
                global_scale_w = tk.max(tk.abs(w)) if mode == "nvfp4" else None
                w_q, scales = tk.quantize(w, mode=mode, global_scale=global_scale_w)
                w_hat = tk.dequantize(
                    w_q,
                    scales,
                    mode=mode,
                    global_scale=global_scale_w,
                    dtype=tk.float32,
                )
                y_q = tk.qqmm(
                    x,
                    w_q,
                    scales,
                    mode=mode,
                    global_scale_x=global_scale_x,
                    global_scale_w=global_scale_w,
                )
                y_hat = x_hat @ tk.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmm(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        dtype = tk.float16 if (tk.default_device() == tk.gpu) else tk.float32
        tests = product(
            [128, 64, 32],  # group_size
            [2, 4, 8],  # bits
            [8, 32, 33, 64],  # M
            [128, 256],  # N
            [128, 256],  # K
            [True, False],  # transposed
        )
        for group_size, bits, M, N, K, transposed in tests:
            with self.subTest(
                shape=(M, N, K),
                group_size=group_size,
                bits=bits,
                transposed=transposed,
            ):
                x = tk.random.normal(shape=(M, K), key=k1) / K**0.5
                w = (
                    tk.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
                    / K**0.5
                )
                x = x.astype(dtype)
                w = w.astype(dtype)
                w_q, scales, biases = tk.quantize(w, group_size, bits)
                w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
                y_q = tk.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )
                y_hat = (x @ w_hat.T) if transposed else (x @ w_hat)
                self.assertEqual(y_q.shape, y_hat.shape)

                tol = 1e-3 if dtype == tk.float32 else 1.5e-3
                self.assertLess((y_q - y_hat).abs().max(), tol)

    def test_qmm_large_dims(self):
        # Regression test for an int16 overflow in the NAX qmm kernels:
        # the per-simdgroup edge sizes were computed as
        # min(SN, short(N - (y_col + tn))), which wraps for distances
        # over 32767 and made store_safe skip a contiguous band of output
        # columns [N - 65536, N - 32768) whenever the M-tile was partial.
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        dtype = tk.float16 if (tk.default_device() == tk.gpu) else tk.float32
        group_size, bits = 64, 4
        K = 128
        tests = [
            (16, 32840),  # unaligned N > 2**15, M < 32: partial M-tile
            (32, 32840),  # M at the small-block dispatch boundary
            (33, 32840),  # unaligned N > 2**15, M % 32 != 0
            (33000, 64),  # M > 2**15: row distance overflows (aligned N)
        ]
        for M, N in tests:
            with self.subTest(shape=(M, N, K)):
                x = tk.random.normal(shape=(M, K), key=k1) / K**0.5
                w = tk.random.normal(shape=(N, K), key=k2) / K**0.5
                x = x.astype(dtype)
                w = w.astype(dtype)
                w_q, scales, biases = tk.quantize(w, group_size, bits)
                w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
                y_q = tk.quantized_matmul(
                    x, w_q, scales, biases, True, group_size, bits
                )
                y_hat = x @ w_hat.T
                self.assertEqual(y_q.shape, y_hat.shape)
                tol = 1e-3 if dtype == tk.float32 else 1.5e-3
                self.assertLess((y_q - y_hat).abs().max(), tol)

    @unittest.skipIf("CI" in os.environ, "too slow in CI")
    def test_qmm_non_transposed(self):
        # The non-transposed matmul (w is [K, N]) is reachable mainly from the
        # vjp of a quantized linear layer, so it gets much less coverage than
        # the transposed one. Sweep it over transformer-sized K/N and over M
        # values that leave a partial M-tile.
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)

        modes = ["mxfp4", "nvfp4", "mxfp8"]
        if tk.default_device() == tk.gpu:
            dtypes = [tk.float16, tk.bfloat16]
        else:
            dtypes = [tk.float32]

        def check_affine(M, K, N, group_size, bits, dtype, batch=()):
            x = tk.random.normal(shape=(*batch, M, K), key=k1, dtype=dtype) / K**0.5
            w = tk.random.normal(shape=(K, N), key=k2, dtype=dtype) / K**0.5
            w_q, scales, biases = tk.quantize(w, group_size, bits)
            w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
            y_q = tk.quantized_matmul(x, w_q, scales, biases, False, group_size, bits)
            y_hat = x @ w_hat
            self.assertEqual(y_q.shape, y_hat.shape)
            tol = 1e-3 if dtype == tk.float32 else 1.5e-3
            self.assertLess((y_q - y_hat).abs().max(), tol)

        def check_fp(M, K, N, mode, dtype, batch=()):
            x = tk.random.normal(shape=(*batch, M, K), key=k1, dtype=dtype) / K**0.5
            w = tk.random.normal(shape=(K, N), key=k2, dtype=dtype) / K**0.5
            w_q, scales = tk.quantize(w, mode=mode)
            w_hat = tk.dequantize(w_q, scales, mode=mode)
            y_q = tk.quantized_matmul(x, w_q, scales, None, False, mode=mode)
            y_hat = x @ w_hat
            self.assertEqual(y_q.shape, y_hat.shape)
            tol = 1e-3 if dtype == tk.float32 else 1.5e-3
            self.assertLess((y_q - y_hat).abs().max(), tol)

        for dtype in dtypes:
            # M sweep. 33..63 is the interesting range: a whole simdgroup of the
            # threadgroup's M-tile falls past the end of the matrix.
            for M in [1, 2, 31, 32, 33, 63, 64, 65, 96, 97, 100, 127, 128, 129]:
                for group_size, bits in [(64, 4), (128, 4), (64, 8)]:
                    with self.subTest(
                        M=M, group_size=group_size, bits=bits, dtype=dtype
                    ):
                        check_affine(M, 512, 1024, group_size, bits, dtype)
                for mode in modes:
                    with self.subTest(M=M, mode=mode, dtype=dtype):
                        check_fp(M, 512, 1024, mode, dtype)

            # Transformer-sized K/N, aligned and unaligned M.
            for K, N in [(2048, 2048), (512, 2048), (2048, 512), (11008, 2048)]:
                for M in [100, 256]:
                    with self.subTest(shape=(M, K, N), dtype=dtype):
                        check_affine(M, K, N, 64, 4, dtype)
                for mode in modes:
                    with self.subTest(shape=(M, K, N), mode=mode, dtype=dtype):
                        check_fp(M, 512, 1024, mode, dtype)

            # Batched x, unaligned M.
            for batch in [(2,), (2, 3)]:
                for M in [33, 250]:
                    with self.subTest(batch=batch, M=M, dtype=dtype):
                        check_affine(M, 512, 1024, 64, 4, dtype, batch=batch)
                for mode in modes:
                    with self.subTest(batch=batch, mode=mode, dtype=dtype):
                        check_fp(M, 512, 1024, mode, dtype, batch=batch)

            # M > 2**15 with a partial M-tile, so the per-simdgroup row count is a
            # distance that does not fit in an int16. Same failure mode as the one
            # test_qmm_large_dims covers for the transposed kernel.
            with self.subTest(shape=(33000, 128, 64), dtype=dtype):
                check_affine(33000, 128, 64, 64, 4, dtype)
                check_fp(33000, 128, 64, mode, dtype)

            # K=64 is the single reduction-tile control; K > 64 spans two or more
            # tiles, which exposed the over-advanced scale pointer.
            for M in [8, 33, 65]:
                for K in [64, 128, 256]:
                    for bits in [2, 4, 8]:
                        with self.subTest(M=M, K=K, bits=bits, dtype=dtype):
                            check_affine(M, K, 128, 32, bits, dtype)
                    for mode in modes:
                        with self.subTest(M=M, K=K, mode=mode, dtype=dtype):
                            check_fp(M, K, 128, mode, dtype)

    def test_qmm_small_m_block(self):
        # The batched and fp-mode variants of the small-M block, which the
        # test_qmm_large_dims shapes cannot reach.
        if tk.default_device() == tk.cpu:
            self.skipTest("Covers GPU kernels only")
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        K = 1024
        tests = [
            # mode, group_size, bits, M, N, batch
            ("affine", 64, 4, 14, 8256, (2,)),  # batched w
            ("mxfp4", None, None, 14, 8256, ()),
        ]
        for mode, group_size, bits, M, N, batch in tests:
            dtype = tk.float16 if mode == "affine" else tk.bfloat16
            with self.subTest(
                mode=mode, group_size=group_size, bits=bits, M=M, N=N, batch=batch
            ):
                x = (tk.random.normal(batch + (M, K), key=k1) / K**0.5).astype(dtype)
                w = (tk.random.normal(batch + (N, K), key=k2) / K**0.5).astype(dtype)
                if mode == "affine":
                    wq = tk.quantize(w, group_size=group_size, bits=bits)
                else:
                    wq = tk.quantize(w, mode=mode)
                w_hat = tk.dequantize(*wq, group_size=group_size, bits=bits, mode=mode)
                y_ref = x @ w_hat.swapaxes(-1, -2)
                y = tk.quantized_matmul(
                    x,
                    *wq,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )
                self.assertLess((y_ref - y).abs().max(), 1e-3)

    def test_qmm_vjp(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)

        bits = 8
        group_size = 64
        M = 64
        N = 1024
        K = 512

        x = tk.random.normal(shape=(2, M, K), key=k1)
        c = tk.ones(shape=(2, M, N))

        transposes = [True, False]
        for transposed in transposes:
            w = tk.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
            w_q, scales, biases = tk.quantize(w, group_size, bits)

            def fn(x):
                return tk.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )

            _, vjp_out = tk.vjp(fn, primals=(x,), cotangents=(c,))

            expected_out = tk.quantized_matmul(
                c, w_q, scales, biases, not transposed, group_size, bits
            )
            self.assertTrue(tk.allclose(vjp_out[0], expected_out))

    def test_qmm_jvp(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)

        bits = 8
        group_size = 64
        M = 64
        N = 128
        K = 128

        x = tk.random.normal(shape=(2, M, K), key=k1)
        x_tan = tk.ones(shape=(2, M, N))

        transposes = [True, False]
        for transposed in transposes:
            w = tk.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
            w_q, scales, biases = tk.quantize(w, group_size, bits)

            def fn(x):
                return tk.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )

            _, jvp_out = tk.jvp(fn, primals=(x,), tangents=(x_tan,))

            expected_out = tk.quantized_matmul(
                x_tan, w_q, scales, biases, transposed, group_size, bits
            )
            self.assertTrue(tk.allclose(jvp_out[0], expected_out))

    def test_qmm_shapes(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        group_size = 64
        bits = 4
        w = tk.random.normal(shape=(32, 256), key=k2)
        w_q, scales, biases = tk.quantize(w, group_size, bits)
        w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
        for s in [(3, 256), (2, 1, 7, 256)]:
            x = tk.random.normal(shape=s, key=k1)
            y_q = tk.quantized_matmul(x, w_q, scales, biases, True, group_size, bits)
            y_hat = x @ w_hat.T
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        w = tk.random.normal(shape=(256, 256), key=k2)
        w_q, scales, biases = tk.quantize(w, group_size, bits)
        w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
        for s in [(3, 256), (2, 1, 7, 256)]:
            x = tk.random.normal(shape=s, key=k1)
            y_q = tk.quantized_matmul(x, w_q, scales, biases, False, group_size, bits)
            y_hat = x @ w_hat
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmv(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 3, 4, 5, 6, 8],  # bits
            [256, 512, 67],  # M
            [64, 256],  # N
            [0, 1, 3, 8],  # B
        )
        for group_size, bits, M, N, B in tests:
            if group_size > N:
                continue
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
                x_shape = (3, 1, N) if B == 0 else (B, 1, N)
                w_shape = (M, N) if B == 0 else (B, M, N)
                x = tk.random.normal(shape=x_shape, key=k1) / N**0.5
                w = tk.random.normal(shape=w_shape, key=k2) / N**0.5
                w_q, scales, biases = tk.quantize(w, group_size, bits)
                w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
                y_q = tk.quantized_matmul(
                    x, w_q, scales, biases, True, group_size, bits
                )
                y_hat = x @ tk.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_fp_qmv(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        tests = product(
            [256, 512, 67, 5, 7],  # M -- 5, 7 exercise out_vec_size < 8 (#3762)
            [64, 256],  # N
            [0, 1, 3, 8],  # B
        )
        modes = ["mxfp4", "nvfp4", "mxfp8"]
        for M, N, B in tests:
            for mode in modes:
                with self.subTest(shape=(B, M, N), mode=mode):
                    x_shape = (3, 1, N) if B == 0 else (B, 1, N)
                    w_shape = (M, N) if B == 0 else (B, M, N)
                    x = tk.random.normal(shape=x_shape, key=k1)
                    w = tk.random.normal(shape=w_shape, key=k2)
                    w_q, scales = tk.quantize(w, mode=mode)
                    w_hat = tk.dequantize(w_q, scales, mode=mode)
                    y_q = tk.quantized_matmul(
                        x,
                        w_q,
                        scales,
                        transpose=True,
                        mode=mode,
                    )
                    y_hat = x @ tk.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test multiple of 16 but not 32
        M = 128
        N = 48
        mode = "nvfp4"
        with self.subTest(shape=(B, M, N), mode=mode):
            x_shape = (1, N)
            w_shape = (M, N)
            x = tk.random.normal(shape=x_shape, key=k1)
            w = tk.random.normal(shape=w_shape, key=k2)
            w_q, scales = tk.quantize(w, mode=mode)
            w_hat = tk.dequantize(w_q, scales, mode=mode)
            y_q = tk.quantized_matmul(
                x,
                w_q,
                scales,
                transpose=True,
                mode=mode,
            )
            y_hat = x @ tk.swapaxes(w_hat, -1, -2)
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_fp_qmv_large_output(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        K = 512
        N = 4096

        for B in [1, 2]:
            with self.subTest(B=B, N=N, K=K):
                x_shape = (1, K) if B == 1 else (B, 1, K)
                w_shape = (N, K) if B == 1 else (B, N, K)
                x = tk.random.normal(shape=x_shape, key=k1) / K**0.5
                w = tk.random.normal(shape=w_shape, key=k2)
                w_q, scales = tk.quantize(w, mode="nvfp4")

                dtypes = (
                    [tk.float16, tk.bfloat16, tk.float32]
                    if tk.default_device() == tk.gpu
                    else [tk.float32]
                )
                for dtype in dtypes:
                    with self.subTest(dtype=dtype):
                        x_t = x.astype(dtype)
                        w_hat = tk.dequantize(w_q, scales, mode="nvfp4", dtype=dtype)
                        y_q = tk.quantized_matmul(
                            x_t,
                            w_q,
                            scales,
                            transpose=True,
                            mode="nvfp4",
                        )
                        y_hat = x_t @ tk.swapaxes(w_hat, -1, -2)
                        self.assertEqual(y_q.shape, y_hat.shape)
                        tol = 1e-2 if dtype == tk.bfloat16 else 1e-3
                        self.assertTrue(tk.allclose(y_q, y_hat, rtol=tol, atol=tol))

    def test_qmv_wide(self):
        # M in [2, vector_limit) routes to qmv_wide -- except K in {64, 128}
        # with power-of-2 bits, which stays on qmv_quad. Check both paths
        # against a dequantize-then-matmul reference, with ragged M (token
        # tail) and ragged N (output-tile remainder). B > 1 stacks a distinct
        # weight matrix per slab and exercises the batched variant.
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        # M <= 9 < vector_limit for these shapes (K, N <= 2048), so all stay on
        # the mat-vec path; 7 and 9 also exercise the token-tail guard.
        Ms = [2, 3, 4, 5, 6, 7, 9]
        Ns = [256, 67]  # 67 is a non-multiple of the 4-row output tile
        Bs = [1, 3]

        # Affine: every bit-width and group size.
        for group_size, bits, K in product(
            [32, 64, 128], [2, 3, 4, 5, 6, 8], [128, 512]
        ):
            for M, N, B in product(Ms, Ns, Bs):
                with self.subTest(M=M, N=N, K=K, B=B, group_size=group_size, bits=bits):
                    x_shape = (M, K) if B == 1 else (B, M, K)
                    w_shape = (N, K) if B == 1 else (B, N, K)
                    x = tk.random.normal(shape=x_shape, key=k1)
                    w = tk.random.normal(shape=w_shape, key=k2)
                    w_q, scales, biases = tk.quantize(w, group_size, bits)
                    w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
                    y_q = tk.quantized_matmul(
                        x, w_q, scales, biases, True, group_size, bits
                    )
                    y_hat = x @ tk.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # FP modes (group_size and bits implied by the mode).
        for mode, K in product(["mxfp4", "nvfp4", "mxfp8"], [128, 512]):
            for M, N, B in product(Ms, Ns, Bs):
                with self.subTest(M=M, N=N, K=K, B=B, mode=mode):
                    x_shape = (M, K) if B == 1 else (B, M, K)
                    w_shape = (N, K) if B == 1 else (B, N, K)
                    x = tk.random.normal(shape=x_shape, key=k1)
                    w = tk.random.normal(shape=w_shape, key=k2)
                    w_q, scales = tk.quantize(w, mode=mode)
                    w_hat = tk.dequantize(w_q, scales, mode=mode)
                    y_q = tk.quantized_matmul(x, w_q, scales, transpose=True, mode=mode)
                    y_hat = x @ tk.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Tiny shapes (M, K, N): small K and non-multiple output rows.
        tiny = [(2, 32, 10), (4, 32, 7), (3, 64, 5), (5, 64, 3)]
        settings = [(4, 32, "affine"), (6, 32, "affine"), (4, 16, "nvfp4")]
        for M, K, N in tiny:
            for bits, group_size, mode in settings:
                with self.subTest(
                    M=M, K=K, N=N, bits=bits, group_size=group_size, mode=mode
                ):
                    x = tk.random.normal(shape=(M, K), key=k1)
                    w = tk.random.normal(shape=(N, K), key=k2)
                    w_q, *sb = tk.quantize(
                        w, group_size=group_size, bits=bits, mode=mode
                    )
                    w_hat = tk.dequantize(
                        w_q, *sb, group_size=group_size, bits=bits, mode=mode
                    )
                    y_q = tk.quantized_matmul(
                        x,
                        w_q,
                        *sb,
                        transpose=True,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )
                    y_hat = x @ tk.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qvm(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 3, 4, 5, 6, 8],  # bits
            [32, 128, 256],  # M
            [128, 256, 67],  # N
            [0, 1, 3, 8],  # B
        )
        for group_size, bits, M, N, B in tests:
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
                if M < group_size:
                    continue
                x_shape = (1, N) if B == 0 else (B, 1, N)
                w_shape = (N, M) if B == 0 else (B, N, M)
                x = tk.random.normal(shape=x_shape, key=k1)
                w = tk.random.normal(shape=w_shape, key=k2)
                w_q, scales, biases = tk.quantize(w, group_size, bits)
                w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
                y_q = tk.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qvm_splitk(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 4, 8],  # bits
            [128],  # M
            [16384],  # N
            [1, 3],  # B
        )
        for group_size, bits, M, N, B in tests:
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
                x_shape = (1, N) if B == 0 else (B, 1, N)
                w_shape = (N, M) if B == 0 else (B, N, M)
                x = 1e-1 * tk.random.normal(shape=x_shape, key=k1)
                w = 1e-1 * tk.random.normal(shape=w_shape, key=k2)
                w_q, scales, biases = tk.quantize(w, group_size, bits)
                w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
                y_q = tk.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 2e-3)

        # Test with 1D vector
        group_size = 32
        bits = 8
        N = 2048
        x = 1e-1 * tk.random.normal(shape=(N,), key=k1)
        w = 1e-1 * tk.random.normal(shape=(N, N), key=k2)
        w_q, scales, biases = tk.quantize(w, group_size, bits)
        w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
        y_q = tk.quantized_matmul(x, w_q, scales, biases, False, group_size, bits)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 2e-3)

    def test_qvm_splitk_multi_row(self):
        # Test qvm split_k with M > 1 to ensure the x row stride is correct
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        tests = product(
            [64, 32],  # group_size
            [4, 8],  # bits
            [128],  # out dim (N)
            [2048, 4096],  # in dim (K) >= 1024 to trigger split_k
            [2, 3],  # M (multiple rows)
        )
        for group_size, bits, N, K, M in tests:
            with self.subTest(M=M, K=K, N=N, group_size=group_size, bits=bits):
                x = 1e-1 * tk.random.normal(shape=(M, K), key=k1)
                w = 1e-1 * tk.random.normal(shape=(K, N), key=k2)
                w_q, scales, biases = tk.quantize(w, group_size, bits)
                w_hat = tk.dequantize(w_q, scales, biases, group_size, bits)
                y_q = tk.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 2e-3)

    def test_fp_qvm(self):
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        tests = product(
            [32, 128, 256],  # M
            [128, 256, 67],  # N
            [0, 1, 3, 8],  # B
        )
        # Add a splitk
        tests = list(tests)
        tests.append((128, 16384, 0))
        modes = ["mxfp4", "nvfp4", "mxfp8"]

        for M, N, B in tests:
            for mode in modes:
                with self.subTest(shape=(B, M, N), mode=mode):
                    x_shape = (1, N) if B == 0 else (B, 1, N)
                    w_shape = (N, M) if B == 0 else (B, N, M)
                    x = tk.random.normal(shape=x_shape, key=k1)
                    w = tk.random.normal(shape=w_shape, key=k2)
                    w_q, scales = tk.quantize(w, mode=mode)
                    w_hat = tk.dequantize(w_q, scales, mode=mode)
                    y_q = tk.quantized_matmul(
                        x,
                        w_q,
                        scales,
                        transpose=False,
                        mode=mode,
                    )
                    y_hat = x @ w_hat
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 2e-3)

    def test_mode_error_cases(self):
        w = tk.random.normal(shape=(256, 256))
        x = tk.random.normal(shape=(1, 256))

        # Invalid mode
        with self.assertRaises(ValueError):
            tk.quantize(w, mode="xyz")

        wq, scales, biases = tk.quantize(w, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            tk.dequantize(wq, scales, biases, bits=4, group_size=32, mode="xyz")

        with self.assertRaises(ValueError):
            tk.quantized_matmul(
                x, wq, scales, biases, bits=4, group_size=32, mode="xyz"
            )

        rhs_indices = tk.array(0)
        with self.assertRaises(ValueError):
            tk.gather_qmm(
                x,
                wq,
                scales,
                biases,
                rhs_indices=rhs_indices,
                bits=4,
                group_size=32,
                mode="xyz",
            )

        # Only quantize floating point types
        with self.assertRaises(ValueError):
            tk.quantize(tk.zeros((128, 128), tk.int32))

        with self.assertRaises(ValueError):
            tk.quantize(tk.zeros((128, 128), tk.int32), mode="mxfp4")

        # Must have bias for affine
        with self.assertRaises(ValueError):
            tk.dequantize(wq, scales, None, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            tk.quantized_matmul(x, wq, scales, None, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            tk.gather_qmm(
                x, wq, scales, None, rhs_indices=rhs_indices, bits=4, group_size=32
            )

        # Must be floating point
        x = tk.zeros(shape=(256,), dtype=tk.int32)
        scales = tk.zeros(scales.shape, dtype=tk.int32)
        biases = tk.zeros(scales.shape, dtype=tk.int32)
        with self.assertRaises(ValueError):
            tk.dequantize(wq, scales, biases, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            tk.quantized_matmul(x, wq, scales, biases, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            tk.gather_qmm(
                x, wq, scales, biases, rhs_indices=rhs_indices, bits=4, group_size=32
            )

    def test_throw(self):
        x = tk.random.normal(shape=(10, 512))
        w = tk.random.normal(shape=(32, 512))
        w_q, scales, biases = tk.quantize(w)

        with self.assertRaises(ValueError):
            tk.quantized_matmul(x, w_q.T, scales, biases)
        with self.assertRaises(ValueError):
            tk.quantized_matmul(x, w_q.T, scales.T, biases)
        with self.assertRaises(ValueError):
            tk.quantized_matmul(x, w_q, scales, biases, False)
        with self.assertRaises(ValueError):
            tk.quantized_matmul(x, w_q, scales.T, biases.T)
        y = tk.quantized_matmul(x, w_q, scales, biases, True)
        tk.eval(y)

    def test_small_matrix(self):
        for w_shape in [(8, 256), (1, 8, 256), (3, 8, 256)]:
            with self.subTest(w_shape=w_shape):
                w = tk.random.normal(shape=(w_shape))
                w_q, scales, biases = tk.quantize(w)
                w_hat = tk.dequantize(w_q, scales, biases)

                # Test qmv
                for shape in [(3, 1, 256), (3, 4, 256)]:
                    x = tk.random.normal(shape=shape)
                    y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=True)
                    y_hat = x @ tk.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

                # Test qmm_t
                x = tk.random.normal(shape=(3, 10, 256))
                y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=True)
                y_hat = x @ tk.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

                # Test qvm
                x = tk.random.normal(shape=(3, 1, 8))
                y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=False)
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

                # Test qmm
                x = tk.random.normal(shape=(3, 10, 8))
                y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=False)
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_non_multiples(self):
        w = tk.random.normal(shape=(33, 256))
        w_q, scales, biases = tk.quantize(w)
        w_hat = tk.dequantize(w_q, scales, biases)

        # Test qmv
        x = tk.random.normal(shape=(1, 256))
        y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm_t
        x = tk.random.normal(shape=(10, 256))
        y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qvm
        x = tk.random.normal(shape=(1, 33))
        y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm
        x = tk.random.normal(shape=(10, 33))
        y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Smaller than 8
        w = tk.random.normal(shape=(3, 256))
        w_q, scales, biases = tk.quantize(w)
        w_hat = tk.dequantize(w_q, scales, biases)

        # Test qmv
        x = tk.random.normal(shape=(1, 256))
        y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm_t
        x = tk.random.normal(shape=(10, 256))
        y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qvm
        x = tk.random.normal(shape=(1, 3))
        y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm
        x = tk.random.normal(shape=(10, 3))
        y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test with larger than 128 unaligned sizes
        w = tk.random.normal(shape=(99, 256))
        w_q, scales, biases = tk.quantize(w)
        w_hat = tk.dequantize(w_q, scales, biases)
        x = tk.random.normal(shape=(129, 256))
        y_q = tk.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmv_small_non_multiples(self):
        # Test very small K and N dimensions (e.g., [MxK] x [NxK].T = [MxN])
        # Each tuple is (M, K, N) representing input rows, weight cols, weight rows
        test_cases = [
            (1, 32, 3),
            (2, 32, 10),
            (1, 32, 5),
            (4, 32, 7),
        ]

        # Test different quantization settings (bits, group_size, mode)
        quantization_settings = [
            (4, 32, "affine"),
            (6, 32, "affine"),
            (4, 16, "nvfp4"),
        ]

        for M, K, N in test_cases:
            for bits, group_size, mode in quantization_settings:
                # Test without batch dimension
                with self.subTest(
                    M=M,
                    K=K,
                    N=N,
                    batch=None,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                ):
                    w = tk.random.normal(shape=(N, K))
                    w_q, *sb = tk.quantize(
                        w,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )
                    w_hat = tk.dequantize(
                        w_q,
                        *sb,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )

                    # Test qmv/qmm_t (transpose=True): [MxK] @ [NxK].T = [MxN]
                    x = tk.random.normal(shape=(M, K))
                    y_q = tk.quantized_matmul(
                        x,
                        w_q,
                        *sb,
                        transpose=True,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )
                    y_hat = x @ tk.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_gather_qmm(self):
        def quantize(w, transpose=True, group_size=None, bits=None, mode="affine"):
            if mode == "affine":
                qw, s, b = tk.quantize(w, group_size=group_size, bits=bits, mode=mode)
            else:
                qw, s = tk.quantize(w, group_size=group_size, bits=bits, mode=mode)
                b = None
            w_hat = tk.dequantize(qw, s, b, group_size=group_size, bits=bits, mode=mode)
            if transpose:
                w_hat = w_hat.swapaxes(-1, -2)
            return w_hat, qw, s, b

        def test_shape(
            M,
            N,
            K,
            dtype=tk.float32,
            batch_A=(),
            batch_B=(),
            lhs_indices=None,
            rhs_indices=None,
            transpose=True,
            group_size=None,
            bits=None,
            mode="affine",
        ):
            with self.subTest(
                M=M,
                N=N,
                K=K,
                dtype=dtype,
                batch_A=batch_A,
                batch_B=batch_B,
                lhs_indices=lhs_indices,
                rhs_indices=rhs_indices,
                transpose=transpose,
                group_size=group_size,
                bits=bits,
                mode=mode,
            ):
                x = tk.random.normal(shape=batch_A + (M, K)).astype(dtype)
                w = tk.random.normal(
                    shape=batch_B + ((N, K) if transpose else (K, N))
                ).astype(dtype)
                w_hat, qw, s, b = quantize(w, transpose, group_size, bits, mode=mode)

                if lhs_indices is not None:
                    lhs_indices = tk.array(lhs_indices)
                if rhs_indices is not None:
                    rhs_indices = tk.array(rhs_indices)

                c1 = tk.gather_mm(x, w_hat, lhs_indices, rhs_indices)
                c2 = tk.gather_qmm(
                    x,
                    qw,
                    s,
                    b,
                    lhs_indices,
                    rhs_indices,
                    transpose=transpose,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )
                self.assertTrue(tk.allclose(c1, c2, atol=1e-4))

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
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
                "mode": "nvfp4",
            },
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
                "mode": "mxfp4",
            },
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
                "mode": "mxfp8",
            },
        )

        for kwargs in inputs:
            test_shape(1, 32, 128, **kwargs)
            test_shape(32, 32, 256, **kwargs)
            test_shape(1, 32, 256, **kwargs)
            test_shape(32, 256, 32, transpose=False, **kwargs)
            test_shape(1, 256, 32, transpose=False, **kwargs)
            test_shape(32, 32, 512, **kwargs)
            test_shape(1, 32, 512, **kwargs)
            test_shape(32, 512, 32, transpose=False, **kwargs)
            test_shape(1, 512, 32, transpose=False, **kwargs)

    def test_gather_qqmm(self):
        if tk.default_device() == tk.cpu:
            self.skipTest("Not implemented for CPU")
            return

        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        batches = (
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
        )
        tests = product(
            batches,
            [1, 32],  # M
            [32, 256],  # N
            [32, 256],  # K
            ["nvfp4", "mxfp8"],  # mode
        )

        for batch, M, N, K, mode in tests:
            with self.subTest(shape=(M, N, K), mode=mode, **batch):
                batch_A, lhs_indices, batch_B, rhs_indices = batch.values()
                x_shape = (*batch_A, M, K)
                w_shape = (*batch_B, N, K)

                x = tk.random.normal(shape=x_shape, key=k1)
                global_scale_x = tk.max(tk.abs(x)) if mode == "nvfp4" else None
                x_hat = tk.dequantize(
                    *tk.quantize(x, mode=mode, global_scale=global_scale_x),
                    mode=mode,
                    dtype=tk.float32,
                    global_scale=global_scale_x,
                )

                w = tk.random.normal(shape=w_shape, key=k2)
                global_scale_w = tk.max(tk.abs(w)) if mode == "nvfp4" else None
                w_q, scales = tk.quantize(w, mode=mode, global_scale=global_scale_w)
                w_hat = tk.dequantize(
                    w_q,
                    scales,
                    mode=mode,
                    global_scale=global_scale_w,
                    dtype=tk.float32,
                )

                if lhs_indices is not None:
                    lhs_indices = tk.array(lhs_indices)
                if rhs_indices is not None:
                    rhs_indices = tk.array(rhs_indices)

                y_q = tk.gather_qqmm(
                    x,
                    w_q,
                    scales,
                    lhs_indices,
                    rhs_indices,
                    mode=mode,
                    global_scale_x=global_scale_x,
                    global_scale_w=global_scale_w,
                )
                y_hat = tk.gather_mm(
                    x_hat, tk.swapaxes(w_hat, -1, -2), lhs_indices, rhs_indices
                )
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmm_fp_type(self):
        indices = tk.array([[2], [0], [1]], dtype=tk.uint32)

        modes = ["mxfp8", "mxfp4"]
        for mode in modes:
            for t in [tk.bfloat16, tk.float16, tk.float32]:
                x = tk.random.normal((32, 256)).astype(t)

                w = tk.random.normal((32, 256))
                wq, s = tk.quantize(w, mode=mode)
                out = tk.quantized_matmul(x, wq, s, mode=mode)
                self.assertEqual(out.dtype, t)

                w = tk.random.normal((4, 32, 256))
                wq, s = tk.quantize(w, mode=mode)

                out = tk.gather_qmm(x, wq, s, rhs_indices=indices, mode=mode)
                self.assertEqual(out.dtype, t)

    def test_gather_matmul_grad(self):
        def quantize(w, transpose=True, group_size=64, bits=4):
            qw, s, b = tk.quantize(w, group_size=group_size, bits=bits)
            w_hat = tk.dequantize(qw, s, b, group_size=group_size, bits=bits)
            if transpose:
                w_hat = w_hat.swapaxes(-1, -2)
            return w_hat, qw, s, b

        lhs_indices = tk.array([[7, 6], [4, 1], [0, 2]], dtype=tk.uint32)
        rhs_indices = tk.array([[2], [0], [1]], dtype=tk.uint32)

        x = tk.random.normal((4, 2, 32, 256))
        w = tk.random.normal((4, 1, 32, 256))
        w_hat, qw, s, b = quantize(w)

        def f_ref(x, w, i1, i2):
            return tk.gather_mm(x, w, i1, i2).sum()

        def f_test(x, qw, s, b, i1, i2):
            return tk.gather_qmm(x, qw, s, b, i1, i2, transpose=True).sum()

        r1 = f_ref(x, w_hat, lhs_indices, rhs_indices)
        r2 = f_test(x, qw, s, b, lhs_indices, rhs_indices)
        self.assertTrue(tk.allclose(r1, r2, atol=1e-4))

        g1 = tk.grad(f_ref)(x, w_hat, lhs_indices, rhs_indices)
        g2 = tk.grad(f_test)(x, qw, s, b, lhs_indices, rhs_indices)
        self.assertTrue(tk.allclose(g1, g2, atol=1e-4))

    def test_gather_qmm_matrix_path(self):
        # Regression test for matrix-size gather_qmm with half precision
        # inputs: on NAX devices the kernel name was built with bk = 32
        # while the kernels are only instantiated with bk = 64, so the
        # kernel lookup failed with "Unable to load kernel".
        key = tk.random.key(0)
        k1, k2 = tk.random.split(key)
        dtype = tk.bfloat16 if (tk.default_device() == tk.gpu) else tk.float32
        E, M, N, K = 4, 64, 512, 512
        lhs_indices = tk.array([0, 1, 2], dtype=tk.uint32)
        rhs_indices = tk.array([0, 2, 3], dtype=tk.uint32)
        for mode in ["affine", "mxfp4"]:
            with self.subTest(mode=mode):
                x = (tk.random.normal(shape=(3, M, K), key=k1) / K**0.5).astype(dtype)
                # Keep w in the same dtype so affine scales do not promote
                # the matmul to float32 (which would skip the NAX route).
                w = tk.random.normal(shape=(E, N, K), key=k2).astype(dtype)
                w_q, *wargs = tk.quantize(w, mode=mode)
                w_hat = tk.dequantize(w_q, *wargs, mode=mode)
                y_q = tk.gather_qmm(
                    x,
                    w_q,
                    *wargs,
                    lhs_indices=lhs_indices,
                    rhs_indices=rhs_indices,
                    transpose=True,
                    mode=mode,
                )
                y_hat = tk.stack(
                    [
                        x[i].astype(tk.float32) @ w_hat[int(rhs_indices[i])].T
                        for i in range(3)
                    ]
                ).astype(dtype)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-1)

    @unittest.skipIf(
        is_m1_mac() and not tk.metal.is_available(),
        "Accelerate bug https://github.com/ml-explore/mlx/pull/3563",
    )
    def test_gather_qmm_sorted(self):
        def quantize(w, transpose=True, group_size=None, mode="affine"):
            if mode == "affine":
                qw, s, b = tk.quantize(w, group_size=group_size, mode=mode)
            else:
                qw, s = tk.quantize(w, mode=mode)
                b = None

            w_hat = tk.dequantize(qw, s, b, group_size=group_size, mode=mode)
            if transpose:
                w_hat = w_hat.swapaxes(-1, -2)
            return w_hat, qw, s, b

        def gather_sort(x, indices):
            N, M = indices.shape
            indices = indices.flatten()
            order = tk.argsort(indices)
            inv_order = tk.argsort(order)
            return x.flatten(0, -3)[order // M], indices[order], inv_order

        def scatter_unsort(x, inv_order, shape=None):
            x = x[inv_order]
            if shape is not None:
                x = tk.unflatten(x, 0, shape)
            return x

        parameters = [
            # L, K, D, E, I, transpose
            (32, 512, 512, 4, 2, True, "affine"),
            (32, 512, 544, 4, 2, True, "mxfp4"),
            (32, 512, 544, 4, 2, True, "nvfp4"),
            (32, 512, 544, 4, 2, True, "mxfp8"),
            (39, 512, 512, 4, 2, True, "affine"),
            (128, 512, 512, 4, 2, True, "affine"),
            (133, 512, 512, 4, 2, True, "affine"),
            (133, 512, 555, 4, 2, True, "affine"),
            (133, 512, 512, 4, 2, True, "affine"),
            (64, 512, 512, 4, 2, False, "affine"),
            (64, 512, 544, 4, 2, False, "mxfp4"),
            (64, 512, 544, 4, 2, False, "nvfp4"),
            (64, 512, 544, 4, 2, False, "mxfp8"),
            (133, 512, 512, 4, 2, False, "affine"),
            (133, 512, 544, 4, 2, False, "affine"),
            (133, 512, 555, 4, 2, False, "affine"),
            (64, 512, 512, 4, 2, False, "affine"),
        ]

        key = tk.random.key(0)
        k1, k2, k3 = tk.random.split(key, 3)
        dtype = tk.float16 if (tk.default_device() == tk.gpu) else tk.float32

        for L, K, D, E, I, transpose, mode in parameters:
            with self.subTest(L=L, K=K, D=D, E=E, I=I, transpose=transpose, mode=mode):
                if mode != "affine":
                    group_size = None
                    dtype = (
                        tk.bfloat16 if (tk.default_device() == tk.gpu) else tk.float32
                    )
                else:
                    group_size = 64
                    dtype = (
                        tk.float16 if (tk.default_device() == tk.gpu) else tk.float32
                    )

                K, D = (K, D) if transpose else (D, K)
                ishape = (L, I)
                xshape = (L, 1, 1, K)
                wshape = (E, D, K) if transpose else (E, K, D)

                indices = (tk.random.uniform(shape=ishape, key=k1) * E).astype(
                    tk.uint32
                )
                x = tk.random.normal(xshape, key=k2) / K**0.5
                w = tk.random.normal(wshape, key=k3) / K**0.5

                x = x.astype(dtype)
                w = w.astype(dtype)

                w, *wq = quantize(
                    w, group_size=group_size, mode=mode, transpose=transpose
                )

                y1 = tk.gather_mm(x, w, rhs_indices=indices)
                y2 = tk.gather_qmm(
                    x,
                    *wq,
                    group_size=group_size,
                    mode=mode,
                    transpose=transpose,
                    rhs_indices=indices,
                )
                xs, idx, inv_order = gather_sort(x, indices)
                y3 = tk.gather_mm(xs, w, rhs_indices=idx, sorted_indices=True)

                y4 = tk.gather_qmm(
                    xs,
                    *wq,
                    group_size=group_size,
                    mode=mode,
                    rhs_indices=idx,
                    transpose=transpose,
                    sorted_indices=True,
                )
                y3 = scatter_unsort(y3, inv_order, indices.shape)
                y4 = scatter_unsort(y4, inv_order, indices.shape)

                tol = 1.5e-5 if (dtype == tk.float32) else 1e-3

                self.assertLess((y1 - y2).abs().max(), tol)
                self.assertLess((y1 - y3).abs().max(), tol)
                self.assertLess((y1 - y4).abs().max(), tol)

                self.assertTrue(tk.allclose(y1, y2, atol=tol))
                self.assertTrue(tk.allclose(y1, y3, atol=tol))
                self.assertTrue(tk.allclose(y1, y4, atol=tol))

    @unittest.skipIf(not tk.metal.is_available(), "requires Metal")
    def test_gather_qmm_sorted_nax_large_m(self):
        E, N, K, group_size = 16, 64, 64, 32
        dtype = tk.float16
        tk.random.seed(0)
        w = (tk.random.normal((E, N, K)) * 0.1).astype(dtype)
        w_q, scales, biases = tk.quantize(w, group_size=group_size, bits=4)
        w_hat = tk.dequantize(w_q, scales, biases, group_size=group_size, bits=4)

        for M in (32767, 32768, 32769, 32832):
            with self.subTest(M=M):
                x = (tk.random.normal((M, 1, K)) * 0.1).astype(dtype)
                rhs_indices = (tk.arange(M) * E // M).astype(tk.uint32)
                y_hat = tk.gather_mm(
                    x.astype(tk.float32),
                    tk.swapaxes(w_hat, -1, -2).astype(tk.float32),
                    rhs_indices=rhs_indices,
                    sorted_indices=True,
                )
                tk.eval(y_hat)
                tk.synchronize()

                for value in (-31.0, 47.0):
                    poison = tk.full(y_hat.shape, value, dtype=tk.float16)
                    tk.eval(poison)
                    tk.synchronize()
                    del poison

                    y_q = tk.gather_qmm(
                        x,
                        w_q,
                        scales,
                        biases,
                        rhs_indices=rhs_indices,
                        transpose=True,
                        group_size=group_size,
                        bits=4,
                        sorted_indices=True,
                    )
                    max_error = (y_q.astype(tk.float32) - y_hat).abs().max()
                    self.assertLess(float(max_error.item()), 5e-2)
                    del y_q, max_error

    @unittest.skipIf(tk.cuda.is_available(), "Not implemented for CUDA")
    def test_gather_qmm_sorted_sliced_weight(self):
        E, R, D, N = 8, 64, 256, 64
        dtype = tk.float16 if (tk.default_device() == tk.gpu) else tk.float32
        tk.random.seed(0)
        w = (tk.random.normal((E, 2 * R, D)) * 0.05).astype(dtype)
        qw, s, b = tk.quantize(w, group_size=64, bits=4)
        x = (tk.random.normal((N, 1, D)) * 0.5).astype(dtype)
        indices = tk.sort(tk.random.randint(0, E, (N,)).astype(tk.uint32))

        for sl in (slice(0, R), slice(R, 2 * R)):
            view = (qw[:, sl], s[:, sl], b[:, sl])
            copy = tuple(tk.contiguous(a) for a in view)
            kwargs = dict(
                rhs_indices=indices,
                transpose=True,
                group_size=64,
                bits=4,
                sorted_indices=True,
            )
            self.assertTrue(
                tk.allclose(
                    tk.gather_qmm(x, *view, **kwargs),
                    tk.gather_qmm(x, *copy, **kwargs),
                    atol=1e-4,
                )
            )

    def test_gather_qmm_grad(self):
        def gather_qmm_ref(x, w, s, b, lhs, rhs, trans, sort):
            if lhs is not None:
                x = x[lhs]
            if rhs is not None:
                w = w[rhs]
                s = s[rhs]
                b = b[rhs]
            return tk.quantized_matmul(x, w, s, b, transpose=trans)

        def gather_qmm(x, w, s, b, lhs, rhs, trans, sort):
            return tk.gather_qmm(
                x,
                w,
                s,
                b,
                transpose=trans,
                lhs_indices=lhs,
                rhs_indices=rhs,
                sorted_indices=sort,
            )

        key = tk.random.key(0)
        k1, k2, k3, k4 = tk.random.split(key, 4)
        dtype = tk.float32

        x = tk.random.normal((16, 1, 256), key=k1).astype(dtype)
        w, s, b = tk.quantize(tk.random.normal((4, 256, 256), key=k2).astype(dtype))
        indices = tk.sort(tk.random.randint(0, 4, shape=(16,), key=k3))
        cotan = tk.random.normal((16, 1, 256), key=k4).astype(dtype)

        (o1,), (dx1, ds1, db1) = tk.vjp(
            lambda x, s, b: gather_qmm_ref(x, w, s, b, None, indices, True, True),
            [x, s, b],
            [cotan],
        )
        (o2,), (dx2, ds2, db2) = tk.vjp(
            lambda x, s, b: gather_qmm(x, w, s, b, None, indices, True, True),
            [x, s, b],
            [cotan],
        )

        self.assertLess((o1 - o2).abs().max(), 1e-4)
        self.assertTrue(tk.allclose(o1, o2, atol=1e-4))
        self.assertTrue(tk.allclose(dx1, dx2, atol=1e-4))
        self.assertTrue(tk.allclose(ds1, ds2, atol=1e-3))
        self.assertTrue(tk.allclose(db1, db2, atol=1e-3))

    def test_vjp_scales_biases(self):
        tk.random.seed(0)
        x = tk.random.normal(shape=(2, 2, 512))
        w = tk.random.normal(shape=(512, 512))
        wq, s, b = tk.quantize(w, bits=4, group_size=64)

        def mm(sb, x, wq):
            return tk.quantized_matmul(x, wq, *sb, bits=4, group_size=64).sum()

        params = (s, b)
        dparams = tk.grad(mm)((s, b), x, wq)

        eps = 8e-3
        # numerical grad check with a few indices
        indices = [(0, 0), (11, 4), (22, 7)]
        for idx in indices:
            for p in [0, 1]:
                params[p][idx] += eps
                out_up = mm(params, x, wq)
                params[p][idx] -= 2 * eps
                out_down = mm(params, x, wq)
                params[p][idx] += eps
                num_ds = (out_up - out_down) / (2 * eps)
                self.assertAlmostEqual(dparams[p][idx], num_ds, delta=2e-2)

    def test_fp_vjp_scales_throws(self):
        tk.random.seed(0)
        x = tk.random.normal(shape=(2, 512))
        w = tk.random.normal(shape=(512, 512))
        for mode in ["mxfp4", "mxfp8", "nvfp4"]:
            wq, s = tk.quantize(w, mode=mode)

            def mm(s, x, wq):
                return tk.quantized_matmul(x, wq, s, mode=mode).sum()

            # Should raise
            with self.assertRaises(ValueError):
                ds = tk.grad(mm)(s, x, wq)

            rhs_indices = tk.array(0)
            with self.assertRaises(ValueError):

                def gmm(s, x, wq):
                    return tk.gather_qmm(
                        x,
                        wq,
                        s,
                        rhs_indices=rhs_indices,
                        mode=mode,
                    ).sum()

                ds = tk.grad(gmm)(s, x, wq)

    def test_quantize_strided(self):
        N = 64
        mode = "nvfp4"
        w = tk.random.normal(shape=(N, N))
        w_q, scales = tk.quantize(w, mode="nvfp4")

        scales = tk.broadcast_to(tk.array(56, tk.uint8), scales.shape)
        w_hat = tk.dequantize(w_q, scales, mode=mode)
        expected = tk.dequantize(w_q, tk.contiguous(scales), mode=mode)
        self.assertTrue(tk.allclose(w_hat, expected))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
