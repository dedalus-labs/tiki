"""Contracts for compiled derivative callbacks without a CUDA launch."""

import unittest
from unittest.mock import patch

import tiki as tk

import compiler


class DerivativeTests(unittest.TestCase):
    # Invariant: an absent tangent contributes zero to the derivative.
    # Witness: multiply with either input fixed, then both inputs active.
    def test_partial_jvp_matches_eager(self):
        def evaluate(compiled, *inputs):
            return compiled.function(*inputs)

        x = tk.array([1.0, 2.0, 3.0])
        y = tk.array([4.0, 5.0, 6.0])
        dx = tk.ones_like(x)
        with patch.object(compiler.Compiled, "launch", evaluate):
            compiled = compiler.compile()(lambda x, y: x * y)
            for function, primal, expected in (
                (lambda a: compiled(a, y), x, y),
                (lambda b: compiled(x, b), y, x),
            ):
                tangent = tk.jvp(function, (primal,), (dx,))[1][0]
                self.assertTrue(tk.array_equal(tangent, expected))
            tangent = tk.jvp(compiled, (x, y), (dx, dx))[1][0]
            self.assertTrue(tk.array_equal(tangent, x + y))

    # Invariant: a single absent tangent follows Tiki's bare-value convention.
    # Witness: square with a None tangent has a zero derivative.
    def test_absent_single_input_tangent_is_zero(self):
        def evaluate(compiled, *inputs):
            return compiled.function(*inputs)

        with patch.object(compiler.Compiled, "launch", evaluate):
            compiled = compiler.compile()(lambda x: x * x)
            x = tk.array([1.0, 2.0, 3.0])
            self.assertTrue(tk.array_equal(compiled._jvp(x, None), tk.zeros_like(x)))

    # Invariant: a VJP that needs a reduction fails before any kernel launch.
    # Witness: vector times scalar needs a scalar cotangent reduction.
    def test_broadcast_vjp_fails_before_launch(self):
        x = tk.ones((3,))
        scalar = tk.array(2.0)
        compiled = compiler.compile()(lambda x, scalar: x * scalar)
        with patch.object(compiler.Compiled, "launch") as launch:
            with self.assertRaisesRegex(
                compiler.UnsupportedDerivativeError, "VJP.*reduction"
            ):
                compiled._vjp((x, scalar), x, x * scalar)
            launch.assert_not_called()

    # Invariant: cooperative forward schedules are not derivative schedules.
    # Witness: transpose and row reduction reject both derivative modes.
    def test_cooperative_derivatives_fail_before_launch(self):
        x = tk.ones((2, 3))
        for schedule, function in (
            (compiler.TransposeSchedule(), lambda x: x.T),
            (compiler.RowSchedule(), lambda x: x + tk.sum(x, axis=-1, keepdims=True)),
        ):
            with self.subTest(schedule=schedule):
                compiled = compiler.compile(schedule=schedule)(function)
                output = function(x)
                with patch.object(compiler.Compiled, "launch") as launch:
                    with self.assertRaisesRegex(
                        compiler.UnsupportedDerivativeError, "derivatives.*elementwise"
                    ):
                        compiled._vjp(x, tk.ones_like(output), output)
                    with self.assertRaisesRegex(
                        compiler.UnsupportedDerivativeError, "derivatives.*elementwise"
                    ):
                        compiled._jvp(x, tk.ones_like(x))
                    launch.assert_not_called()

    # Invariant: supported derivative shapes retain their numerical contract.
    # Witness: equal-shape VJPs and scalar-broadcast JVPs of multiply.
    def test_elementwise_derivatives_match_eager(self):
        def evaluate(compiled, *inputs):
            return compiled.function(*inputs)

        with patch.object(compiler.Compiled, "launch", evaluate):
            compiled = compiler.compile()(lambda x, y: x * y)
            for shape in ((), (3,)):
                x = tk.full(shape, 2.0)
                y = tk.full(shape, 3.0)
                cotangent = tk.full(shape, 4.0)
                gradients = tk.vjp(compiled, (x, y), (cotangent,))[1]
                for got, expected in zip(gradients, (y * cotangent, x * cotangent)):
                    self.assertTrue(tk.array_equal(got, expected))
            x = tk.array([1.0, 2.0, 3.0])
            scalar = tk.array(2.0)
            _, tangents = tk.jvp(
                compiled, (x, scalar), (tk.ones_like(x), tk.array(1.0))
            )
            self.assertTrue(tk.array_equal(tangents[0], scalar + x))


if __name__ == "__main__":
    unittest.main()
