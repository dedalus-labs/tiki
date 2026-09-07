# Copyright © 2026 Dedalus Labs, Inc.

"""A custom vmap rule owns the batched computation.

Invariant: vmapping a custom_function never vmaps the recorded forward graph,
so a forward built from primitives without a vmap (an opaque kernel, or
as_strided here) is vectorized by its rule alone. Witness: the rule's result
is used and the unbatched path is unchanged."""

import unittest

import mlx.core as mx


class TestCustomVmapOpaque(unittest.TestCase):
    def test_rule_replaces_the_recorded_forward(self) -> None:
        @mx.custom_function
        def reverse(x: mx.array) -> mx.array:
            return mx.as_strided(x, shape=x.shape, strides=(-1,), offset=x.size - 1)

        @reverse.vmap
        def _(inputs: mx.array, axes: int) -> tuple[mx.array, int]:
            return mx.moveaxis(inputs, axes, 0)[:, ::-1], 0

        batch = mx.arange(12, dtype=mx.float32).reshape(3, 4)
        self.assertTrue(mx.array_equal(reverse(batch[1]), batch[1][::-1]).item())
        self.assertTrue(mx.array_equal(mx.vmap(reverse)(batch), batch[:, ::-1]).item())
        self.assertTrue(
            mx.array_equal(mx.vmap(reverse, in_axes=1)(batch), batch.T[:, ::-1]).item()
        )

    def test_default_rule_still_vmaps_the_function(self) -> None:
        doubled = mx.custom_function(lambda x: x * 2)
        batch = mx.ones((2, 3))
        self.assertTrue(mx.array_equal(mx.vmap(doubled)(batch), batch * 2).item())


if __name__ == "__main__":
    unittest.main()
