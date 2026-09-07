# Copyright © 2026 Dedalus Labs, Inc.

import tiki as tk
import tiki.layout as tl
import tiki_tests
import numpy as np
from tiki.layout.tensor import (
    expand,
    from_array,
    realize,
    right_major_layout,
    squeeze,
    unsqueeze,
)


def values(array: tk.array) -> np.ndarray:
    tk.eval(array)
    return np.asarray(array)


class TestArrayEngine(tiki_tests.TIKITestCase):
    def test_from_array_normalizes_strided_vectors(self) -> None:
        for array in (tk.arange(8)[::2], tk.arange(8)[::-1]):
            np.testing.assert_array_equal(
                values(realize(from_array(array))), values(array)
            )

    def test_engine_checks_addresses_without_negative_index_wrapping(self) -> None:
        engine = tl.ArrayEngine(tk.arange(4))
        for offset in (-1, 4, 1.5):
            with self.assertRaises(tl.LayoutError):
                engine[offset]
        with self.assertRaises(tl.LayoutError):
            tl.ArrayEngine(tk.arange(4), -1)

    def test_realize_rejects_addresses_outside_the_engine(self) -> None:
        base = tk.arange(4, dtype=tk.float32)
        for offset, layout in (
            (0, tl.Layout(4, 100)),
            (0, tl.Layout(4, -1)),
            (3, tl.Layout(2, 1)),
        ):
            with self.subTest(offset=offset, layout=str(layout)):
                with self.assertRaises(tl.LayoutError):
                    realize(tl.Tensor(tl.ArrayEngine(base, offset), layout))

    def test_realize_does_not_change_the_stride_algebra(self) -> None:
        layout = tl.Layout((2, 2), (tl.F2(1), tl.F2(1)))
        with self.assertRaises(tl.LayoutError):
            realize(tl.Tensor(tl.ArrayEngine(tk.arange(4)), layout))

    def test_composed_tensor_indexing_preserves_parent_addresses(self) -> None:
        layout = tl.ComposedLayout(tl.Swizzle(2, 0, 2), 0, tl.Layout((4, 4), (4, 1)))
        tensor = tl.Tensor(tl.ArrayEngine(tk.arange(16)), layout)
        for row in range(4):
            for column in range(4):
                self.assertEqual(tensor[row, column], layout(row, column))
                self.assertEqual(tensor[row, None][column], layout(row, column))

    # Invariant: from_array pairs the flattened storage with the dense
    # right-major layout, and tensor[coordinate] reads engine[layout(coordinate)].
    # Witness: arange(12) as 3x4, element (1, 2) is 6.
    def test_from_array(self):
        tensor = from_array(tk.arange(12, dtype=tk.float32).reshape(3, 4))
        self.assertEqual(tensor.layout, tl.Layout((3, 4), (4, 1)))
        self.assertEqual(tensor[1, 2], 6.0)
        self.assertEqual(
            right_major_layout((2, 3, 5)), tl.Layout((2, 3, 5), (15, 5, 1))
        )

    # Invariant: realize is a zero-copy view whose values equal the layout's
    # map over the engine, for dense, transposed, offset, hierarchical, and
    # negative-stride layouts.
    # Witness: arange(12) under each layout, compared element by element.
    def test_realize_views(self):
        array = tk.arange(12, dtype=tk.float32)
        flat = values(array)
        cases = [
            (0, tl.Layout((3, 4), (4, 1))),
            (0, tl.Layout((4, 3), (1, 4))),
            (1, tl.Layout(3, 4)),
            (0, tl.Layout(((3, 2), 2), ((4, 1), 2))),
            (3, tl.Layout(4, -1)),
        ]
        for offset, layout in cases:
            tensor = tl.Tensor(tl.ArrayEngine(array, offset), layout)
            view = values(realize(tensor))
            expected = np.array(
                [flat[offset + layout(index)] for index in range(tl.size(layout))]
            )
            self.assertEqual(
                view.shape, tuple(int(extent) for extent in tl.flatten(layout.shape))
            )
            np.testing.assert_array_equal(
                view.reshape(-1, order="F"), expected, err_msg=str(layout)
            )

    # Invariant: the round trip through from_array and realize returns the
    # original values, and a transposed layout realizes as the transpose.
    # Witness: a 3x4 array and its transpose.
    def test_round_trip(self):
        array = tk.arange(12, dtype=tk.float32).reshape(3, 4)
        tensor = from_array(array)
        np.testing.assert_array_equal(values(realize(tensor)), values(array))
        transposed = tl.Tensor(tensor.accessor, tl.Layout((4, 3), (1, 4)))
        np.testing.assert_array_equal(values(realize(transposed)), values(array.T))

    # Invariant: a composed layout cannot be a view.
    # Witness: a swizzled 4x4 layout.
    def test_realize_rejects_composed(self):
        engine = tl.ArrayEngine(tk.arange(16, dtype=tk.float32))
        composed = tl.ComposedLayout(tl.Swizzle(2, 0, 2), 0, tl.Layout((4, 4), (4, 1)))
        with self.assertRaises(tl.LayoutError):
            realize(tl.Tensor(engine, composed))


class TestBroadcast(tiki_tests.TIKITestCase):
    def test_axis_operations_preserve_hierarchical_modes(self) -> None:
        layout = tl.Layout(((2, 3), 4), ((1, 2), 6))
        tensor = tl.Tensor(tl.ArrayEngine(tk.arange(24)), layout)
        inserted = unsqueeze(tensor, 2)
        self.assertEqual(inserted.layout, tl.Layout(((2, 3), 4, 1), ((1, 2), 6, 0)))
        self.assertEqual(squeeze(inserted, 2).layout, layout)
        expanded = expand(inserted, (6, 4, 2))
        self.assertEqual(expanded.layout, tl.Layout(((2, 3), 4, 2), ((1, 2), 6, 0)))

    # Invariant (zop): unsqueeze inserts an extent-1 stride-0 mode, expand
    # follows the trailing-axis rule with stride 0 on expanded axes, and squeeze
    # removes an extent-1 mode; none of them allocate.
    # Witness: a bias of 4 broadcast against a 3x4 activation.
    def test_bias_broadcast(self):
        bias = from_array(tk.array([1.0, 2.0, 3.0, 4.0]))
        with_batch = unsqueeze(bias, 0)
        self.assertEqual(with_batch.layout, tl.Layout((1, 4), (0, 1)))
        expanded = expand(bias, (3, 4))
        self.assertEqual(expanded.layout, tl.Layout((3, 4), (0, 1)))
        np.testing.assert_array_equal(
            values(realize(expanded)), np.tile([1.0, 2.0, 3.0, 4.0], (3, 1))
        )
        self.assertEqual(squeeze(with_batch, 0).layout, bias.layout)
        self.assertIs(expanded.accessor, bias.accessor)

    # Invariant: an extent that cannot be proven compatible is a LayoutError.
    # Witness: squeezing an extent of 4, expanding 4 to 5, and expanding to
    # fewer axes.
    def test_incompatible_extents_raise(self):
        bias = from_array(tk.array([1.0, 2.0, 3.0, 4.0]))
        with self.assertRaises(tl.LayoutError):
            squeeze(bias, 0)
        with self.assertRaises(tl.LayoutError):
            expand(bias, (5,))
        with self.assertRaises(tl.LayoutError):
            expand(unsqueeze(bias, 0), (4,))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
