"""Contracts for compiled derivative callbacks without a CUDA launch."""

import unittest
from unittest.mock import patch

import mlx.core as mx

import tiki as tk


class DerivativeTests(unittest.TestCase):
    # Invariant: an absent tangent contributes zero to the derivative.
    # Witness: multiply with either input fixed, then both inputs active.
    def test_partial_jvp_matches_eager(self):
        def evaluate(compiled, *inputs):
            return compiled.function(*inputs)

        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([4.0, 5.0, 6.0])
        dx = mx.ones_like(x)
        with patch.object(tk.Compiled, "launch", evaluate):
            compiled = tk.compile()(lambda x, y: x * y)
            for function, primal, expected in (
                (lambda a: compiled(a, y), x, y),
                (lambda b: compiled(x, b), y, x),
            ):
                tangent = mx.jvp(function, (primal,), (dx,))[1][0]
                self.assertTrue(mx.array_equal(tangent, expected))
            tangent = mx.jvp(compiled, (x, y), (dx, dx))[1][0]
            self.assertTrue(mx.array_equal(tangent, x + y))

    # Invariant: a single absent tangent follows MLX's bare-value convention.
    # Witness: square with a None tangent has a zero derivative.
    def test_absent_single_input_tangent_is_zero(self):
        def evaluate(compiled, *inputs):
            return compiled.function(*inputs)

        with patch.object(tk.Compiled, "launch", evaluate):
            compiled = tk.compile()(lambda x: x * x)
            x = mx.array([1.0, 2.0, 3.0])
            self.assertTrue(mx.array_equal(compiled._jvp(x, None), mx.zeros_like(x)))

    # Invariant: a VJP that needs a reduction fails before any kernel launch.
    # Witness: vector times scalar needs a scalar cotangent reduction.
    def test_broadcast_vjp_fails_before_launch(self):
        x = mx.ones((3,))
        scalar = mx.array(2.0)
        compiled = tk.compile()(lambda x, scalar: x * scalar)
        with patch.object(tk.Compiled, "launch") as launch:
            with self.assertRaisesRegex(
                tk.UnsupportedDerivativeError, "VJP.*reduction"
            ):
                compiled._vjp((x, scalar), x, x * scalar)
            launch.assert_not_called()

    # Invariant: cooperative forward schedules are not derivative schedules.
    # Witness: transpose and row reduction reject both derivative modes.
    def test_cooperative_derivatives_fail_before_launch(self):
        x = mx.ones((2, 3))
        for schedule, function in (
            (tk.TransposeSchedule(), lambda x: x.T),
            (tk.RowSchedule(), lambda x: x + mx.sum(x, axis=-1, keepdims=True)),
        ):
            with self.subTest(schedule=schedule):
                compiled = tk.compile(schedule=schedule)(function)
                output = function(x)
                with patch.object(tk.Compiled, "launch") as launch:
                    with self.assertRaisesRegex(
                        tk.UnsupportedDerivativeError, "derivatives.*elementwise"
                    ):
                        compiled._vjp(x, mx.ones_like(output), output)
                    with self.assertRaisesRegex(
                        tk.UnsupportedDerivativeError, "derivatives.*elementwise"
                    ):
                        compiled._jvp(x, mx.ones_like(x))
                    launch.assert_not_called()

    # Invariant: supported derivative shapes retain their numerical contract.
    # Witness: equal-shape VJPs and scalar-broadcast JVPs of multiply.
    def test_elementwise_derivatives_match_eager(self):
        def evaluate(compiled, *inputs):
            return compiled.function(*inputs)

        with patch.object(tk.Compiled, "launch", evaluate):
            compiled = tk.compile()(lambda x, y: x * y)
            for shape in ((), (3,)):
                x = mx.full(shape, 2.0)
                y = mx.full(shape, 3.0)
                cotangent = mx.full(shape, 4.0)
                gradients = mx.vjp(compiled, (x, y), (cotangent,))[1]
                for got, expected in zip(gradients, (y * cotangent, x * cotangent)):
                    self.assertTrue(mx.array_equal(got, expected))
            x = mx.array([1.0, 2.0, 3.0])
            scalar = mx.array(2.0)
            _, tangents = mx.jvp(
                compiled, (x, scalar), (mx.ones_like(x), mx.array(1.0))
            )
            self.assertTrue(mx.array_equal(tangents[0], scalar + x))


if __name__ == "__main__":
    unittest.main()
