# Copyright © 2026 Dedalus Labs, Inc.

import mlx.core as mx
import mlx.tiki as tk
import mlx_tests
import numpy as np
from mlx.tiki.tensor import (
    expand,
    from_array,
    realize,
    right_major_layout,
    squeeze,
    unsqueeze,
)


def values(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.asarray(array)


class TestArrayEngine(mlx_tests.MLXTestCase):
    # Invariant: from_array pairs the flattened storage with the dense
    # right-major layout, and tensor[coordinate] reads engine[layout(coordinate)].
    # Witness: arange(12) as 3x4, element (1, 2) is 6.
    def test_from_array(self):
        tensor = from_array(mx.arange(12, dtype=mx.float32).reshape(3, 4))
        self.assertEqual(tensor.layout, tk.Layout((3, 4), (4, 1)))
        self.assertEqual(tensor[1, 2], 6.0)
        self.assertEqual(
            right_major_layout((2, 3, 5)), tk.Layout((2, 3, 5), (15, 5, 1))
        )

    # Invariant: realize is a zero-copy view whose values equal the layout's
    # map over the engine, for dense, transposed, offset, hierarchical, and
    # negative-stride layouts.
    # Witness: arange(12) under each layout, compared element by element.
    def test_realize_views(self):
        array = mx.arange(12, dtype=mx.float32)
        flat = values(array)
        cases = [
            (0, tk.Layout((3, 4), (4, 1))),
            (0, tk.Layout((4, 3), (1, 4))),
            (1, tk.Layout(3, 4)),
            (0, tk.Layout(((3, 2), 2), ((4, 1), 2))),
            (3, tk.Layout(4, -1)),
        ]
        for offset, layout in cases:
            tensor = tk.Tensor(tk.ArrayEngine(array, offset), layout)
            view = values(realize(tensor))
            expected = np.array(
                [flat[offset + layout(index)] for index in range(tk.size(layout))]
            )
            self.assertEqual(
                view.shape, tuple(int(extent) for extent in tk.flatten(layout.shape))
            )
            np.testing.assert_array_equal(
                view.reshape(-1, order="F"), expected, err_msg=str(layout)
            )

    # Invariant: the round trip through from_array and realize returns the
    # original values, and a transposed layout realizes as the transpose.
    # Witness: a 3x4 array and its transpose.
    def test_round_trip(self):
        array = mx.arange(12, dtype=mx.float32).reshape(3, 4)
        tensor = from_array(array)
        np.testing.assert_array_equal(values(realize(tensor)), values(array))
        transposed = tk.Tensor(tensor.accessor, tk.Layout((4, 3), (1, 4)))
        np.testing.assert_array_equal(values(realize(transposed)), values(array.T))

    # Invariant: a composed layout cannot be a view.
    # Witness: a swizzled 4x4 layout.
    def test_realize_rejects_composed(self):
        engine = tk.ArrayEngine(mx.arange(16, dtype=mx.float32))
        composed = tk.ComposedLayout(tk.Swizzle(2, 0, 2), 0, tk.Layout((4, 4), (4, 1)))
        with self.assertRaises(tk.LayoutError):
            realize(tk.Tensor(engine, composed))


class TestBroadcast(mlx_tests.MLXTestCase):
    # Invariant (zop): unsqueeze inserts an extent-1 stride-0 mode, expand
    # follows the trailing-axis rule with stride 0 on expanded axes, and squeeze
    # removes an extent-1 mode; none of them allocate.
    # Witness: a bias of 4 broadcast against a 3x4 activation.
    def test_bias_broadcast(self):
        bias = from_array(mx.array([1.0, 2.0, 3.0, 4.0]))
        with_batch = unsqueeze(bias, 0)
        self.assertEqual(with_batch.layout, tk.Layout((1, 4), (0, 1)))
        expanded = expand(bias, (3, 4))
        self.assertEqual(expanded.layout, tk.Layout((3, 4), (0, 1)))
        np.testing.assert_array_equal(
            values(realize(expanded)), np.tile([1.0, 2.0, 3.0, 4.0], (3, 1))
        )
        self.assertEqual(squeeze(with_batch, 0).layout, bias.layout)
        self.assertIs(expanded.accessor, bias.accessor)

    # Invariant: an extent that cannot be proven compatible is a LayoutError.
    # Witness: squeezing an extent of 4, expanding 4 to 5, and expanding to
    # fewer axes.
    def test_incompatible_extents_raise(self):
        bias = from_array(mx.array([1.0, 2.0, 3.0, 4.0]))
        with self.assertRaises(tk.LayoutError):
            squeeze(bias, 0)
        with self.assertRaises(tk.LayoutError):
            expand(bias, (5,))
        with self.assertRaises(tk.LayoutError):
            expand(unsqueeze(bias, 0), (4,))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
