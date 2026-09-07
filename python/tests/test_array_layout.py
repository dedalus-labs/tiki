# Copyright © 2026 Dedalus Labs, Inc.

import tiki as tk
import tiki_tests


class TestArrayLayout(tiki_tests.TIKITestCase):
    # Invariant: strides and offset describe the evaluated storage map, so an
    # as_strided round trip reproduces the view and a transpose reports the
    # swapped strides.
    # Witness: a 3x4 float32 array, its transpose, and an offset slice.
    def test_strides_and_offset(self):
        a = tk.arange(12, dtype=tk.float32).reshape(3, 4)
        self.assertEqual(a.strides, (4, 1))
        self.assertEqual(a.offset, 0)
        t = a.T
        self.assertEqual(t.strides, (1, 4))
        view = a[1:, 2:]
        self.assertEqual(view.strides, (4, 1))
        self.assertEqual(view.offset, 6)
        rebuilt = tk.as_strided(
            a.reshape(-1), shape=view.shape, strides=view.strides, offset=view.offset
        )
        self.assertTrue(tk.array_equal(rebuilt, view))

    # Invariant: a zero-size or scalar array has consistent metadata.
    # Witness: a 0-d array and an empty one.
    def test_degenerate(self):
        self.assertEqual(tk.array(1.0).strides, ())
        self.assertEqual(tk.zeros((0, 3)).strides, (3, 1))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
