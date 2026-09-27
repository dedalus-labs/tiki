# Copyright © 2026 Dedalus Labs, Inc.

"""A custom vmap rule owns the batched computation.

Invariant: vmapping a custom_function never vmaps the recorded forward graph,
so a forward built from primitives without a vmap (an opaque kernel, or
as_strided here) is vectorized by its rule alone. Witness: the rule's result
is used and the unbatched path is unchanged."""

import unittest

import tiki as tk


class TestCustomVmapOpaque(unittest.TestCase):
    def test_rule_replaces_the_recorded_forward(self) -> None:
        @tk.custom_function
        def reverse(x: tk.array) -> tk.array:
            return tk.as_strided(x, shape=x.shape, strides=(-1,), offset=x.size - 1)

        @reverse.vmap
        def _(inputs: tk.array, axes: int) -> tuple[tk.array, int]:
            return tk.moveaxis(inputs, axes, 0)[:, ::-1], 0

        batch = tk.arange(12, dtype=tk.float32).reshape(3, 4)
        self.assertTrue(tk.array_equal(reverse(batch[1]), batch[1][::-1]).item())
        self.assertTrue(tk.array_equal(tk.vmap(reverse)(batch), batch[:, ::-1]).item())
        self.assertTrue(
            tk.array_equal(tk.vmap(reverse, in_axes=1)(batch), batch.T[:, ::-1]).item()
        )

    def test_default_rule_still_vmaps_the_function(self) -> None:
        doubled = tk.custom_function(lambda x: x * 2)
        batch = tk.ones((2, 3))
        self.assertTrue(tk.array_equal(tk.vmap(doubled)(batch), batch * 2).item())


if __name__ == "__main__":
    unittest.main()
