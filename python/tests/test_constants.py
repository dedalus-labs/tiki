# Copyright © 2023 Apple Inc.

import unittest

import numpy as np
import tiki as tk
import tiki_tests


class TestConstants(tiki_tests.TIKITestCase):
    def test_constants_values(self):
        # Check if tiki constants match expected values
        self.assertAlmostEqual(
            tk.e, 2.71828182845904523536028747135266249775724709369995
        )
        self.assertAlmostEqual(
            tk.euler_gamma, 0.5772156649015328606065120900824024310421
        )
        self.assertAlmostEqual(tk.inf, float("inf"))
        self.assertTrue(np.isnan(tk.nan))
        self.assertIsNone(tk.newaxis)
        self.assertAlmostEqual(tk.pi, 3.1415926535897932384626433)

    def test_constants_availability(self):
        # Check if tiki constants are available
        self.assertTrue(hasattr(tk, "e"))
        self.assertTrue(hasattr(tk, "euler_gamma"))
        self.assertTrue(hasattr(tk, "inf"))
        self.assertTrue(hasattr(tk, "nan"))
        self.assertTrue(hasattr(tk, "newaxis"))
        self.assertTrue(hasattr(tk, "pi"))

    def test_newaxis_for_reshaping_arrays(self):
        arr_1d = tk.array([1, 2, 3, 4, 5])
        arr_2d_column = arr_1d[:, tk.newaxis]
        expected_result = tk.array([[1], [2], [3], [4], [5]])
        self.assertTrue(tk.array_equal(arr_2d_column, expected_result))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
