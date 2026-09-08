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
        row_major = tk.Layout((4, 8), stride=(8, 1))
        column_major = tk.Layout((4, 8), stride=(1, 4))
        blocked = tk.Layout(((4, 2), (2, 4)), stride=((1, 32), (4, 8)))
        self.assertEqual(row_major(2, 3), 19)
        self.assertEqual(column_major(2, 3), 14)
        self.assertEqual(blocked.shape, ((4, 2), (2, 4)))
        self.assertEqual(format(row_major, "cute"), "(4, 8):(8, 1)")
        self.assertEqual(
            tk.logical_divide(tk.Layout(24, stride=1), tk.Layout(4, stride=2)),
            tk.Layout((4, (2, 3)), stride=(2, (1, 8))),
        )

    # Invariant: the algebra reproduces the CuTe documentation results.
    # Witness: the canonical divide, complement, composition, coalesce,
    # inverse, nullspace, and zipped-divide examples.
    def test_algebra_matches_cute_documentation(self):
        self.assertEqual(
            tk.complement(tk.Layout(4, stride=2), 24), tk.Layout((2, 3), stride=(1, 8))
        )
        self.assertEqual(
            tk.compose(
                tk.Layout((6, 2), stride=(8, 2)), tk.Layout((4, 3), stride=(3, 1))
            ),
            tk.Layout(((2, 2), 3), stride=((24, 2), 8)),
        )
        self.assertEqual(
            tk.coalesce(tk.Layout((2, (1, 6)), stride=(1, (6, 2)))),
            tk.Layout(12, stride=1),
        )
        self.assertEqual(
            tk.right_inverse(tk.Layout((4, 8), stride=(8, 1))),
            tk.Layout((8, 4), stride=(4, 1)),
        )
        self.assertEqual(
            tk.nullspace(tk.Layout((4, 8), stride=(0, 1))), tk.Layout(4, stride=1)
        )
        self.assertEqual(
            tk.zipped_divide(
                tk.Layout((8, 8), stride=(1, 8)), tk.Layout((2, 4), stride=(1, 2))
            ),
            tk.Layout(((2, 4), 8), stride=((1, 2), 8)),
        )
        layout = tk.Layout((4, 8), stride=(8, 1))
        self.assertEqual(
            (tk.size(layout), tk.cosize(layout), tk.rank(layout), tk.depth(layout)),
            (32, 32, 2, 1),
        )

    # Invariant: natural coordinates are colexicographic, so the three
    # coordinate forms of a layout agree.
    # Witness: (3,(2,4)):(2,(1,6)) evaluated as 17, (2,5), and (2,(1,2)).
    def test_coordinate_forms_agree(self):
        layout = tk.Layout((3, (2, 4)), stride=(2, (1, 6)))
        self.assertEqual(layout(17), layout(2, 5))
        self.assertEqual(layout(17), layout(2, (1, 2)))
        self.assertEqual(tk.idx2crd(19, (4, 8)), (3, 4))

    # Invariant: a failed precondition raises LayoutError, a ValueError, and
    # never returns a weaker layout.
    # Witness: left inverse and complement of (4,4):(1,2), whose modes overlap.
    def test_preconditions_raise_layout_error(self):
        overlapping = tk.Layout((4, 4), stride=(1, 2))
        with self.assertRaises(tk.LayoutError):
            tk.left_inverse(overlapping)
        with self.assertRaises(tk.LayoutError):
            tk.complement(overlapping)
        self.assertTrue(issubclass(tk.LayoutError, ValueError))


class TestComposedLayout(mlx_tests.MLXTestCase):
    def test_integer_slice_offsets_cannot_erase_xor_addition(self) -> None:
        layout = tk.Layout((2, 2), stride=(tk.F2(1), tk.F2(1)))
        with self.assertRaises(tk.LayoutError):
            tk.slice_and_offset((1, None), layout)

    def test_public_swizzle_constructor_enforces_its_contract(self) -> None:
        for args in ((1, 0, 0), (2, 0, 1), (-1, 0, 1), (1, -1, 1), (1.5, 0, 2)):
            with self.subTest(args=args), self.assertRaises(tk.LayoutError):
                tk.Swizzle(**dict(zip(("bits", "base", "shift"), args)))

    # Invariant: str and repr of a swizzle or layout are the keyword constructor
    # call, composed layouts one component per line in evaluation order, and
    # format(value, "cute") is CuTe's notation. Any other format code fails.
    # Witness: the guide's swizzled 4 by 4 layout.
    def test_prints_the_keyword_constructor(self):
        layout = tk.Layout((4, 4), stride=(4, 1))
        swizzle = tk.Swizzle(bits=2, base=0, shift=2)
        composed = layout.swizzle(swizzle)
        self.assertEqual(repr(layout), "Layout(shape=(4, 4), stride=(4, 1))")
        self.assertEqual(str(layout), repr(layout))
        self.assertEqual(format(layout, "cute"), "(4, 4):(4, 1)")
        self.assertEqual(repr(swizzle), "Swizzle(bits=2, base=0, shift=2)")
        self.assertEqual(str(swizzle), repr(swizzle))
        self.assertEqual(format(swizzle, "cute"), "SW_2_0_2")
        self.assertEqual(
            str(composed),
            "ComposedLayout(\n"
            "    inner=Layout(shape=(4, 4), stride=(4, 1)),\n"
            "    offset=0,\n"
            "    outer=Swizzle(bits=2, base=0, shift=2),\n"
            ")",
        )
        self.assertEqual(repr(composed), str(composed))
        with self.assertRaises(ValueError):
            format(layout, "hex")

    # Invariant: parameters that share a type are keyword-only, so every call
    # site names them. Witness: each positional form raises TypeError.
    def test_constructors_take_keywords(self):
        with self.assertRaises(TypeError):
            tk.Swizzle(2, 0, 2)
        with self.assertRaises(TypeError):
            tk.Layout((4, 4), (4, 1))
        with self.assertRaises(TypeError):
            tk.ComposedLayout(tk.Swizzle(bits=2, base=0, shift=2), 0, tk.Layout((4, 4)))

    # Invariant: describe lists every leaf coordinate by position with its
    # extent and stride, then the index formula; a composed layout wraps the
    # formula in its outer transform and shows a nonzero offset.
    # Witness: the 2 by 2 tiling of a 4 by 4 row-major layout, that layout
    # under a swizzle with offset 4, and a rank-one layout.
    def test_describe_lists_one_coordinate_per_line(self):
        base = tk.Layout((4, 4), stride=(4, 1))
        self.assertEqual(
            tk.logical_divide(base, (2, 2)).describe(),
            "coordinate  extent  stride\n"
            "c[0][0]     2       4\n"
            "c[0][1]     2       8\n"
            "c[1][0]     2       1\n"
            "c[1][1]     2       2\n"
            "index = 4 * c[0][0] + 8 * c[0][1] + 1 * c[1][0] + 2 * c[1][1]",
        )
        self.assertEqual(
            base.swizzle(tk.Swizzle(bits=2, base=0, shift=2), offset=4).describe(),
            "coordinate  extent  stride\n"
            "c[0]        4       4\n"
            "c[1]        4       1\n"
            "index = Swizzle(bits=2, base=0, shift=2)(4 + 4 * c[0] + 1 * c[1])",
        )
        self.assertEqual(
            tk.Layout(4, stride=1).describe(),
            "coordinate  extent  stride\nc           4       1\nindex = 1 * c",
        )

    def swizzled(self):
        return tk.ComposedLayout(
            outer=tk.Swizzle(bits=2, base=0, shift=2),
            offset=0,
            inner=tk.Layout((4, 4), stride=(4, 1)),
        )

    # Invariant: a composed layout evaluates outer(offset + inner(coordinate)),
    # formats in CuTe notation as outer o {offset} o inner, and has no stride.
    # Witness: Swizzle(bits=2,base=0,shift=2) over (4,4):(4,1) at (1,2): 6 -> 7.
    def test_evaluates_and_formats(self):
        composed = self.swizzled()
        self.assertEqual(composed(1, 2), 7)
        self.assertEqual(format(composed, "cute"), "SW_2_0_2 o {0} o (4, 4):(4, 1)")
        self.assertEqual(composed.shape, (4, 4))
        with self.assertRaises(tk.LayoutError):
            composed.stride
        with self.assertRaises(tk.LayoutError):
            tk.ComposedLayout(
                outer=tk.Swizzle(bits=1, base=0, shift=1),
                offset=-1,
                inner=tk.Layout((), stride=()),
            )()

    # Invariant (CUTLASS): a swizzle whose bit fields overlap is rejected,
    # because it is not a permutation. PyCuTe and zop's bootstrap both accept it.
    # Witness: Swizzle(bits=1, base=0, shift=0) maps 0 and 1 to 0; Swizzle(bits=1, base=0, shift=1) is fine.
    def test_overlapping_swizzle_is_rejected(self):
        with self.assertRaises(tk.LayoutError):
            tk.ComposedLayout(
                outer=tk.Swizzle(bits=1, base=0, shift=0),
                offset=0,
                inner=tk.Layout(2, stride=1),
            )
        with self.assertRaises(tk.LayoutError):
            tk.check_swizzle(tk.Swizzle(bits=2, base=0, shift=1))
        self.assertEqual(tk.check_swizzle(tk.Swizzle(bits=1, base=0, shift=1)).shift, 1)
        from mlx.tiki._pycute import Swizzle as ReferenceSwizzle

        self.assertEqual({ReferenceSwizzle(1, 0, 0)(index) for index in range(2)}, {0})

    # Invariant (zop): parent(fixed, free) == engine_delta + residual(free) at
    # every coordinate, for affine, swizzled, and nested-swizzled layouts. An
    # affine slice moves the fixed contribution outside; a composed slice keeps
    # it inside and reports zero displacement.
    # Witness: fixing row 1 of each 4x4 layout, all four columns.
    def test_slicing_preserves_every_address(self):
        affine = tk.Layout((4, 4), stride=(4, 1))
        swizzled = self.swizzled()
        nested = tk.ComposedLayout(
            outer=tk.Swizzle(bits=1, base=0, shift=1), offset=0, inner=swizzled
        )
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
                outer=tk.Swizzle(bits=bits, base=base, shift=shift),
                offset=0,
                inner=tk.Layout(shape, stride=stride),
            )
            theirs = tl.ComposedLayout(
                tl.Swizzle(bits=bits, base=base, shift=shift),
                tl.Layout(shape, stride=stride),
                offset=0,
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
        tensor = tk.make_tensor(tk.Layout((4, 4), stride=(4, 1)))
        tensor[1, 2] = 42.0
        self.assertEqual(tensor[1, 2], 42.0)
        self.assertEqual(tensor.accessor[tensor.layout(1, 2)], 42.0)
        self.assertEqual(tk.identity_tensor((3, 4))[1, 2], (1, 2))
        self.assertTrue(isinstance(tensor.accessor, tk.MutableEngine))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
