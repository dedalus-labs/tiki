# Copyright © 2026 Dedalus Labs, Inc.

"""``array.is_tracer`` says whether an array can be evaluated.

Invariant: it is True only for arrays derived from a vmap or compile
placeholder, which have no storage; inside a derivative rule the primals can
be evaluated and it is False, even when that rule runs under vmap only if the
array itself descends from the vmap placeholder. Witness: the flag observed
from inside a vjp rule under grad, under vmap, and under vmap of grad."""

import unittest

import tiki as tk


class TestArrayIsTracer(unittest.TestCase):
    def test_placeholders_are_the_only_tracers(self) -> None:
        seen: list[bool] = []

        @tk.custom_function
        def double(x: tk.array) -> tk.array:
            return x * 2

        @double.vjp
        def _(primals: tk.array, cotangent: tk.array, output: tk.array) -> tk.array:
            seen.append(primals.is_tracer)
            return cotangent * 2

        x = tk.ones((2, 3))
        self.assertFalse(x.is_tracer)
        tk.grad(lambda x: double(x).sum())(x)
        self.assertEqual(seen, [False])
        tk.vmap(lambda a: seen.append(a.is_tracer) or a)(x)
        self.assertEqual(seen, [False, True])
        tk.vmap(lambda a: tk.grad(lambda b: double(b).sum())(a))(x)
        self.assertEqual(seen, [False, True, True])


if __name__ == "__main__":
    unittest.main()
