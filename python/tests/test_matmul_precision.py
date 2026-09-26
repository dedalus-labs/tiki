# Copyright © 2026 Dedalus Labs, Inc.

"""Float32 matmul on the GPU is float32 unless TF32 is requested.

Invariant: without MLX_ENABLE_TF32 the GPU matmul error against a float64
reference is at the single-precision level, on CUDA through cuBLAS and on
Metal through the M5 neural-accelerator kernels. Witness: a 256x256 product,
where TF32 gives an error of 2e-2 (GH200) or 5e-2 (M5) and float32 2e-5."""

import unittest

import mlx.core as mx


@unittest.skipUnless(mx.default_device() == mx.gpu, "measures the GPU matmul")
class TestMatmulPrecision(unittest.TestCase):
    def test_float32_matmul_is_not_tf32_by_default(self) -> None:
        a = mx.random.normal((256, 256), key=mx.random.key(9))
        product = a @ a
        mx.eval(a, product)
        with mx.stream(mx.cpu):
            reference = a.astype(mx.float64) @ a.astype(mx.float64)
            error = mx.abs(product.astype(mx.float64) - reference).max().item()
        self.assertLess(error, 1e-3)


if __name__ == "__main__":
    unittest.main()
