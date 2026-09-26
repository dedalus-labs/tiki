# Copyright © 2026 Dedalus Labs, Inc.

import itertools
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


def cases():
    for n in (1, 2, 3, 4, 5, 7):
        for start, stop, step in itertools.product(
            (None, 0, 1, -1, -2), (None, 1, 2, -1, n), (1, 2, 3, -1, -2)
        ):
            yield (n,), (slice(start, stop, step),)
    for shape in ((2, 3), (3, 2), (5, 2)):
        for step in (2, 3, -2):
            yield shape, (slice(None, None, step), slice(None))
            yield shape, (slice(None), slice(None, None, step))


class TestSliceAutograd(mlx_tests.MLXTestCase):
    # Invariant: the VJP and JVP of a strided slice touch exactly the sliced
    # positions, including when the slice selects a single element.
    # Witness: every (start, stop, step) on lengths 1 to 7 and strided rows and
    # columns of small matrices, checked against NumPy indexing.
    def test_strided_slice_vjp_jvp(self):
        rng = np.random.default_rng(0)
        for shape, idx in cases():
            x = rng.normal(size=shape).astype(np.float32)
            selected = x[idx]
            if selected.size == 0:
                continue
            cotangent = rng.normal(size=selected.shape).astype(np.float32)
            tangent = rng.normal(size=shape).astype(np.float32)
            expected_vjp = np.zeros(shape, np.float32)
            expected_vjp[idx] = cotangent
            expected_jvp = tangent[idx]

            f = lambda a: a[idx]
            with self.subTest(
                shape=shape, idx=[(part.start, part.stop, part.step) for part in idx]
            ):
                vjp = mx.vjp(f, (mx.array(x),), (mx.array(cotangent),))[1][0]
                jvp = mx.jvp(f, (mx.array(x),), (mx.array(tangent),))[1][0]
                self.assertEqual(vjp.shape, expected_vjp.shape)
                self.assertEqual(jvp.shape, expected_jvp.shape)
                self.assertTrue(np.allclose(np.asarray(vjp), expected_vjp))
                self.assertTrue(np.allclose(np.asarray(jvp), expected_jvp))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
