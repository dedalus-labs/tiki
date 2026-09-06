# Copyright © 2026 Dedalus Labs, Inc.

import importlib.util
import unittest

import mlx.tiki as tk
import mlx_tests

HAS_TENSOR_LAYOUTS = importlib.util.find_spec("tensor_layouts") is not None


class TestLayout(mlx_tests.MLXTestCase):
    # Invariant: the zop layouts.md examples hold verbatim.
    # Witness: row-major, column-major, and blocked layouts from that page.
    def test_zop_examples(self):
        row_major = tk.Layout((4, 8), (8, 1))
        column_major = tk.Layout((4, 8), (1, 4))
        blocked = tk.Layout(((4, 2), (2, 4)), ((1, 32), (4, 8)))
        self.assertEqual(row_major(2, 3), 19)
        self.assertEqual(column_major(2, 3), 14)
        self.assertEqual(blocked.shape, ((4, 2), (2, 4)))
        self.assertEqual(str(row_major), "(4, 8):(8, 1)")
        self.assertEqual(
            tk.logical_divide(tk.Layout(24, 1), tk.Layout(4, 2)),
            tk.Layout((4, (2, 3)), (2, (1, 8))),
        )

    # Invariant: the algebra reproduces the CuTe documentation results.
    # Witness: the canonical divide, complement, composition, coalesce,
    # inverse, nullspace, and zipped-divide examples.
    def test_algebra_matches_cute_documentation(self):
        self.assertEqual(tk.complement(tk.Layout(4, 2), 24), tk.Layout((2, 3), (1, 8)))
        self.assertEqual(
            tk.compose(tk.Layout((6, 2), (8, 2)), tk.Layout((4, 3), (3, 1))),
            tk.Layout(((2, 2), 3), ((24, 2), 8)),
        )
        self.assertEqual(
            tk.coalesce(tk.Layout((2, (1, 6)), (1, (6, 2)))), tk.Layout(12, 1)
        )
        self.assertEqual(
            tk.right_inverse(tk.Layout((4, 8), (8, 1))), tk.Layout((8, 4), (4, 1))
        )
        self.assertEqual(tk.nullspace(tk.Layout((4, 8), (0, 1))), tk.Layout(4, 1))
        self.assertEqual(
            tk.zipped_divide(tk.Layout((8, 8), (1, 8)), tk.Layout((2, 4), (1, 2))),
            tk.Layout(((2, 4), 8), ((1, 2), 8)),
        )
        layout = tk.Layout((4, 8), (8, 1))
        self.assertEqual(
            (tk.size(layout), tk.cosize(layout), tk.rank(layout), tk.depth(layout)),
            (32, 32, 2, 1),
        )

    # Invariant: natural coordinates are colexicographic, so the three
    # coordinate forms of a layout agree.
    # Witness: (3,(2,4)):(2,(1,6)) evaluated as 17, (2,5), and (2,(1,2)).
    def test_coordinate_forms_agree(self):
        layout = tk.Layout((3, (2, 4)), (2, (1, 6)))
        self.assertEqual(layout(17), layout(2, 5))
        self.assertEqual(layout(17), layout(2, (1, 2)))
        self.assertEqual(tk.idx2crd(19, (4, 8)), (3, 4))

    # Invariant: a failed precondition raises LayoutError, a ValueError, and
    # never returns a weaker layout.
    # Witness: left inverse and complement of (4,4):(1,2), whose modes overlap.
    def test_preconditions_raise_layout_error(self):
        overlapping = tk.Layout((4, 4), (1, 2))
        with self.assertRaises(tk.LayoutError):
            tk.left_inverse(overlapping)
        with self.assertRaises(tk.LayoutError):
            tk.complement(overlapping)
        self.assertTrue(issubclass(tk.LayoutError, ValueError))


class TestComposedLayout(mlx_tests.MLXTestCase):
    def swizzled(self):
        return tk.ComposedLayout(tk.Swizzle(2, 0, 2), 0, tk.Layout((4, 4), (4, 1)))

    # Invariant: a composed layout evaluates outer(offset + inner(coordinate)),
    # prints as outer o {offset} o inner, and has no stride.
    # Witness: Swizzle(2,0,2) over (4,4):(4,1) at (1,2): 6 -> 7.
    def test_evaluates_and_formats(self):
        composed = self.swizzled()
        self.assertEqual(composed(1, 2), 7)
        self.assertEqual(str(composed), "SW_2_0_2 o {0} o (4, 4):(4, 1)")
        self.assertEqual(composed.shape, (4, 4))
        with self.assertRaises(tk.LayoutError):
            composed.stride
        with self.assertRaises(tk.LayoutError):
            tk.ComposedLayout(tk.Swizzle(1, 0, 1), -1, tk.Layout((), ()))()

    # Invariant (zop): parent(fixed, free) == engine_delta + residual(free) at
    # every coordinate, for affine, swizzled, and nested-swizzled layouts. An
    # affine slice moves the fixed contribution outside; a composed slice keeps
    # it inside and reports zero displacement.
    # Witness: fixing row 1 of each 4x4 layout, all four columns.
    def test_slicing_preserves_every_address(self):
        affine = tk.Layout((4, 4), (4, 1))
        swizzled = self.swizzled()
        nested = tk.ComposedLayout(tk.Swizzle(1, 0, 1), 0, swizzled)
        for parent in (affine, swizzled, nested):
            residual, delta = tk.slice_and_offset((1, None), parent)
            for column in range(4):
                self.assertEqual(
                    delta + residual(column),
                    parent(1, column),
                    f"{parent} column {column}",
                )
        self.assertEqual(tk.slice_and_offset((1, None), affine)[1], 4)
        self.assertEqual(tk.slice_and_offset((1, None), swizzled)[1], 0)

    # Invariant: composed evaluation agrees with the independent tensor-layouts
    # reference at every coordinate, including after slicing.
    # Witness: three swizzles over row-major 8x8 and 4x4 inner layouts.
    @unittest.skipUnless(HAS_TENSOR_LAYOUTS, "tensor-layouts is not installed")
    def test_tensor_layouts_cross_check(self):
        import tensor_layouts as tl

        cases = [
            ((2, 0, 2), (4, 4), (4, 1)),
            ((3, 0, 3), (8, 8), (8, 1)),
            ((1, 1, 2), (8, 8), (8, 1)),
        ]
        for (bits, base, shift), shape, stride in cases:
            ours = tk.ComposedLayout(
                tk.Swizzle(bits, base, shift), 0, tk.Layout(shape, stride)
            )
            theirs = tl.ComposedLayout(
                tl.Swizzle(bits, base, shift), tl.Layout(shape, stride), offset=0
            )
            for row in range(shape[0]):
                for column in range(shape[1]):
                    self.assertEqual(
                        ours(row, column),
                        theirs((row, column)),
                        f"{ours} at {(row, column)}",
                    )


class TestTensor(mlx_tests.MLXTestCase):
    # Invariant: tensor[coordinate] == engine[layout(coordinate)], and the
    # identity tensor returns its own coordinates without storage.
    # Witness: a 4x4 owned tensor written at (1,2) and the identity at (1,2).
    def test_engine_composed_with_layout(self):
        tensor = tk.make_tensor(tk.Layout((4, 4), (4, 1)))
        tensor[1, 2] = 42.0
        self.assertEqual(tensor[1, 2], 42.0)
        self.assertEqual(tensor.accessor[tensor.layout(1, 2)], 42.0)
        self.assertEqual(tk.identity_tensor((3, 4))[1, 2], (1, 2))
        self.assertTrue(isinstance(tensor.accessor, tk.MutableEngine))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
