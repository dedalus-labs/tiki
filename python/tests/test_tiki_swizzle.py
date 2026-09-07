# Copyright © 2026 Dedalus Labs, Inc.

"""Index transforms preserve one layout contract across storage and composition."""

import unittest

import mlx.core as mx
import mlx.tiki as tk


class TestSwizzle(unittest.TestCase):
    def test_value_is_immutable_and_hashes_by_parameters(self) -> None:
        first = tk.Swizzle(2, 0, 2)
        second = tk.Swizzle(2, 0, 2)
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        for field in ("bits", "base", "shift"):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(first, field, 0)
        self.assertEqual(first(6), 7)

    def test_invalid_indices_and_parameters_raise_layout_errors(self) -> None:
        for value in (True, 1.5, -1, 2**63):
            with self.subTest(value=value), self.assertRaises(tk.LayoutError):
                tk.Swizzle(2, 0, 2)(value)
        for args in (
            (2, 0, 1),
            (2, 0, -1),
            (1, 62, 1),
            (0, 0, -(2**63)),
            (2**64, 0, 2),
        ):
            with self.subTest(args=args), self.assertRaises(tk.LayoutError):
                tk.Swizzle(*args)

    def test_signed_field_directions_are_involutions(self) -> None:
        for shift in (-3, 3):
            transform = tk.Swizzle(2, 1, shift)
            for value in range(256):
                self.assertEqual(transform(transform(value)), value)

    def test_composition_preserves_the_inner_domain_and_internal_offset(self) -> None:
        base = tk.Layout((4, 4), (4, 1))
        transform = tk.Swizzle(2, 0, 2)
        composed = tk.compose(transform, base, offset=4)
        self.assertEqual(composed, base.swizzle(transform, offset=4))
        self.assertIs(composed.outer, transform)
        self.assertIs(composed.inner, base)
        self.assertEqual(composed.shape, base.shape)
        for row in range(4):
            for column in range(4):
                self.assertEqual(
                    composed(row, column), transform(4 + base(row, column))
                )
        with self.assertRaises(tk.LayoutError):
            tk.compose(transform, base, offset=0.5)
        with self.assertRaises(tk.LayoutError):
            base.swizzle(base)
        with self.assertRaises(tk.LayoutError):
            composed.swizzle(base)

    def test_affine_and_nonlinear_outer_maps_compose_in_order(self) -> None:
        base = tk.Layout((4, 4), (4, 1))
        first = base.swizzle(tk.Swizzle(2, 0, 2))
        for outer in (tk.Layout(32, 2), tk.Swizzle(1, 0, 1), first):
            composed = tk.compose(outer, first, offset=1)
            for row in range(4):
                for column in range(4):
                    self.assertEqual(
                        composed(row, column), outer(1 + first(row, column))
                    )

    def test_algebra_results_preserve_the_layout_interface(self) -> None:
        layouts = (
            tk.logical_divide(tk.Layout(16), tk.Layout(4)),
            tk.zipped_divide(tk.Layout((8, 8)), tk.Layout((2, 4))),
            tk.coalesce(tk.Layout((2, 8))),
            tk.complement(tk.Layout(4, 2), 24),
            tk.make_layout([tk.Layout(4), tk.Layout(4, 4)]),
            tk.Layout((4, 4))[0],
        )
        transform = tk.Swizzle(2, 0, 2)
        for layout in layouts:
            with self.subTest(layout=str(layout)):
                self.assertIsInstance(layout, tk.Layout)
                self.assertEqual(layout.swizzle(transform)(0), transform(layout(0)))

    def test_slicing_retains_the_engine_and_exact_address_map(self) -> None:
        engine = tk.ArrayEngine(mx.arange(16))
        layout = tk.Layout((4, 4), (4, 1)).swizzle(tk.Swizzle(2, 0, 2))
        tensor = tk.Tensor(engine, layout)
        for row in range(4):
            sliced = tensor[row, None]
            self.assertIs(sliced.accessor.base, engine.base)
            for column in range(4):
                self.assertEqual(sliced[column], tensor[row, column])
        with self.assertRaises(tk.LayoutError):
            tk.realize(tensor)

    def test_stride_only_operations_reject_nonlinear_maps(self) -> None:
        layout = tk.Layout((4, 4)).swizzle(tk.Swizzle(2, 0, 2))
        with self.assertRaises(tk.LayoutError):
            tk.coalesce(layout)
        with self.assertRaises(tk.LayoutError):
            tk.logical_divide(layout, tk.Layout(4))
        with self.assertRaises(tk.LayoutError):
            tk.cosize(layout)


if __name__ == "__main__":
    unittest.main()
