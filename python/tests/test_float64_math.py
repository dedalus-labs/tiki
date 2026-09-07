# Copyright © 2026 Dedalus Labs, Inc.

"""Float64 transcendentals on the CPU carry double precision.

Invariant: exp, sin, cos, and erf of a float64 array agree with the C library
to a few ulps. Witness: a sweep that includes large arguments, where a
single-precision polynomial loses everything past the seventh digit."""

import math
import unittest

import mlx.core as mx


class TestFloat64Math(unittest.TestCase):
    def test_transcendentals_are_double_precision(self) -> None:
        values = [0.0, 1e-9, 0.3, 1.2345678901234567, 12.5, -7.75, 100.0, 1e6 + 0.5]
        with mx.stream(mx.cpu):
            x = mx.array(values, dtype=mx.float64)
            for name, function, reference in (
                ("exp", mx.exp, math.exp),
                ("sin", mx.sin, math.sin),
                ("cos", mx.cos, math.cos),
                ("erf", mx.erf, math.erf),
            ):
                got = function(x).tolist()
                for value, result in zip(values, got):
                    expected = reference(value)
                    with self.subTest(function=name, value=value):
                        self.assertAlmostEqual(
                            result, expected, delta=1e-13 * max(1.0, abs(expected))
                        )


if __name__ == "__main__":
    unittest.main()
