# Copyright © 2026 Dedalus Labs, Inc.

"""A derivative rule may evaluate its primals without severing an outer transformation.

Invariant: forward mode retains the graph while a rule runs, as reverse mode
does, so a nested jvp through a rule that evaluates is the true second
derivative. Witness: d/dx of jvp(x^2) with a rule that calls eval; the result
is 2, and was 0 when the eval detached the inner primal from the outer tracer."""

import unittest

import tiki as tk


class TestJvpRetainsGraph(unittest.TestCase):
    def test_nested_jvp_through_an_evaluating_rule(self) -> None:
        @tk.custom_function
        def square(x: tk.array) -> tk.array:
            return x * x

        @square.jvp
        def _(primals: tk.array, tangents: tk.array) -> tk.array:
            tk.eval(primals)
            return 2 * primals * tangents

        x = tk.array([1.0, 2.0, 3.0])
        ones = tk.ones(3)
        first = lambda x: tk.jvp(square, [x], [ones])[1][0]
        self.assertEqual(tk.jvp(first, [x], [ones])[1][0].tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(tk.grad(lambda x: first(x).sum())(x).tolist(), [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
