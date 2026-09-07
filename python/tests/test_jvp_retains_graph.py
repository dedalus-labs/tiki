# Copyright © 2026 Dedalus Labs, Inc.

"""A derivative rule may evaluate its primals without severing an outer transformation.

Invariant: forward mode retains the graph while a rule runs, as reverse mode
does, so a nested jvp through a rule that evaluates is the true second
derivative. Witness: d/dx of jvp(x^2) with a rule that calls eval; the result
is 2, and was 0 when the eval detached the inner primal from the outer tracer."""

import unittest

import mlx.core as mx


class TestJvpRetainsGraph(unittest.TestCase):
    def test_nested_jvp_through_an_evaluating_rule(self) -> None:
        @mx.custom_function
        def square(x: mx.array) -> mx.array:
            return x * x

        @square.jvp
        def _(primals: mx.array, tangents: mx.array) -> mx.array:
            mx.eval(primals)
            return 2 * primals * tangents

        x = mx.array([1.0, 2.0, 3.0])
        ones = mx.ones(3)
        first = lambda x: mx.jvp(square, [x], [ones])[1][0]
        self.assertEqual(mx.jvp(first, [x], [ones])[1][0].tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(mx.grad(lambda x: first(x).sum())(x).tolist(), [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
