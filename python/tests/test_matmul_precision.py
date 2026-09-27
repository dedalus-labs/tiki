# Copyright © 2026 Dedalus Labs, Inc.

"""Float32 matmul on the GPU is float32 unless TF32 is requested.

Invariant: without TIKI_ENABLE_TF32 the GPU matmul error against a float64
reference is at the single-precision level, on CUDA through cuBLAS and on
Metal through the M5 neural-accelerator kernels. Witness: a 256x256 product,
where TF32 gives an error of 2e-2 (GH200) or 5e-2 (M5) and float32 2e-5."""

import os
import subprocess
import sys
import textwrap
import unittest

import tiki as tk


@unittest.skipUnless(tk.default_device() == tk.gpu, "measures the GPU matmul")
class TestMatmulPrecision(unittest.TestCase):
    def test_float32_matmul_is_not_tf32_by_default(self) -> None:
        # The default must hold despite the runner's override. Use a fresh process
        # for the 256x256 witness because Tiki caches the setting on first use.
        env = os.environ.copy()
        env.pop("TIKI_ENABLE_TF32", None)
        subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""\
                    import unittest

                    import tiki as tk

                    tk.set_default_device(tk.gpu)
                    a = tk.random.normal((256, 256), key=tk.random.key(9))
                    product = a @ a
                    tk.eval(a, product)
                    with tk.stream(tk.cpu):
                        reference = a.astype(tk.float64) @ a.astype(tk.float64)
                        error = tk.abs(product.astype(tk.float64) - reference).max().item()
                    unittest.TestCase().assertLess(error, 1e-3)
                    """),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
