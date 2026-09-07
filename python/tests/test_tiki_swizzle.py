# Copyright © 2026 Dedalus Labs, Inc.

"""Index transforms preserve one layout contract across storage and composition."""

import unittest

import tiki as tk
import tiki.layout as tl


class TestSwizzle(unittest.TestCase):
    def test_value_is_immutable_and_hashes_by_parameters(self) -> None:
        first = tl.Swizzle(2, 0, 2)
        second = tl.Swizzle(2, 0, 2)
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        for field in ("bits", "base", "shift"):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(first, field, 0)
        self.assertEqual(first(6), 7)

    def test_invalid_indices_and_parameters_raise_layout_errors(self) -> None:
        for value in (True, 1.5, -1, 2**63):
            with self.subTest(value=value), self.assertRaises(tl.LayoutError):
                tl.Swizzle(2, 0, 2)(value)
        for args in (
            (2, 0, 1),
            (2, 0, -1),
            (1, 62, 1),
            (0, 0, -(2**63)),
            (2**64, 0, 2),
        ):
            with self.subTest(args=args), self.assertRaises(tl.LayoutError):
                tl.Swizzle(*args)

    def test_signed_field_directions_are_involutions(self) -> None:
        for shift in (-3, 3):
            transform = tl.Swizzle(2, 1, shift)
            for value in range(256):
                self.assertEqual(transform(transform(value)), value)

    def test_composition_preserves_the_inner_domain_and_internal_offset(self) -> None:
        base = tl.Layout((4, 4), (4, 1))
        transform = tl.Swizzle(2, 0, 2)
        composed = tl.compose(transform, base, offset=4)
        self.assertEqual(composed, base.swizzle(transform, offset=4))
        self.assertIs(composed.outer, transform)
        self.assertIs(composed.inner, base)
        self.assertEqual(composed.shape, base.shape)
        for row in range(4):
            for column in range(4):
                self.assertEqual(
                    composed(row, column), transform(4 + base(row, column))
                )
        with self.assertRaises(tl.LayoutError):
            tl.compose(transform, base, offset=0.5)
        with self.assertRaises(tl.LayoutError):
            base.swizzle(base)
        with self.assertRaises(tl.LayoutError):
            composed.swizzle(base)

    def test_affine_and_nonlinear_outer_maps_compose_in_order(self) -> None:
        base = tl.Layout((4, 4), (4, 1))
        first = base.swizzle(tl.Swizzle(2, 0, 2))
        for outer in (tl.Layout(32, 2), tl.Swizzle(1, 0, 1), first):
            composed = tl.compose(outer, first, offset=1)
            for row in range(4):
                for column in range(4):
                    self.assertEqual(
                        composed(row, column), outer(1 + first(row, column))
                    )

    def test_algebra_results_preserve_the_layout_interface(self) -> None:
        layouts = (
            tl.logical_divide(tl.Layout(16), tl.Layout(4)),
            tl.zipped_divide(tl.Layout((8, 8)), tl.Layout((2, 4))),
            tl.coalesce(tl.Layout((2, 8))),
            tl.complement(tl.Layout(4, 2), 24),
            tl.make_layout([tl.Layout(4), tl.Layout(4, 4)]),
            tl.Layout((4, 4))[0],
        )
        transform = tl.Swizzle(2, 0, 2)
        for layout in layouts:
            with self.subTest(layout=str(layout)):
                self.assertIsInstance(layout, tl.Layout)
                self.assertEqual(layout.swizzle(transform)(0), transform(layout(0)))

    def test_slicing_retains_the_engine_and_exact_address_map(self) -> None:
        engine = tl.ArrayEngine(tk.arange(16))
        layout = tl.Layout((4, 4), (4, 1)).swizzle(tl.Swizzle(2, 0, 2))
        tensor = tl.Tensor(engine, layout)
        for row in range(4):
            sliced = tensor[row, None]
            self.assertIs(sliced.accessor.base, engine.base)
            for column in range(4):
                self.assertEqual(sliced[column], tensor[row, column])
        with self.assertRaises(tl.LayoutError):
            tl.realize(tensor)

    def test_stride_only_operations_reject_nonlinear_maps(self) -> None:
        layout = tl.Layout((4, 4)).swizzle(tl.Swizzle(2, 0, 2))
        with self.assertRaises(tl.LayoutError):
            tl.coalesce(layout)
        with self.assertRaises(tl.LayoutError):
            tl.logical_divide(layout, tl.Layout(4))
        with self.assertRaises(tl.LayoutError):
            tl.cosize(layout)


if __name__ == "__main__":
    unittest.main()
