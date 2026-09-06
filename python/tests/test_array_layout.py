# Copyright © 2026 Dedalus Labs, Inc.

import mlx.core as mx
import mlx_tests


class TestArrayLayout(mlx_tests.MLXTestCase):
    # Invariant: strides and offset describe the evaluated storage map, so an
    # as_strided round trip reproduces the view and a transpose reports the
    # swapped strides.
    # Witness: a 3x4 float32 array, its transpose, and an offset slice.
    def test_strides_and_offset(self):
        a = mx.arange(12, dtype=mx.float32).reshape(3, 4)
        self.assertEqual(a.strides, (4, 1))
        self.assertEqual(a.offset, 0)
        t = a.T
        self.assertEqual(t.strides, (1, 4))
        view = a[1:, 2:]
        self.assertEqual(view.strides, (4, 1))
        self.assertEqual(view.offset, 6)
        rebuilt = mx.as_strided(
            a.reshape(-1), shape=view.shape, strides=view.strides, offset=view.offset
        )
        self.assertTrue(mx.array_equal(rebuilt, view))

    # Invariant: a zero-size or scalar array has consistent metadata.
    # Witness: a 0-d array and an empty one.
    def test_degenerate(self):
        self.assertEqual(mx.array(1.0).strides, ())
        self.assertEqual(mx.zeros((0, 3)).strides, (3, 1))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
