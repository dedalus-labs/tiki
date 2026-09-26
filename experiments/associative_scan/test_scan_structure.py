"""A scan preserves pytree identity independently of traversal order."""

import unittest
from typing import NamedTuple, TypedDict

import numpy as np
import tiki as tk

from scan import associative_scan


class Pair(NamedTuple):
    a: tk.array
    b: tk.array


LiteralTree = TypedDict(
    "LiteralTree", {"a.b": tk.array, "a": dict[str, tk.array], "empty": tuple[()]}
)


class TestScanStructure(unittest.TestCase):
    def test_combine_sees_the_original_axis_order(self) -> None:
        matrices = (
            np.random.default_rng(5).integers(-1, 2, size=(5, 2, 2)).astype(np.float32)
        )
        elems = tk.array(matrices.transpose(1, 0, 2))

        def combine(left: tk.array, right: tk.array) -> tk.array:
            return tk.einsum("itk,ktj->itj", left, right)

        expected = np.stack(
            [
                (
                    np.linalg.multi_dot(list(matrices[:prefix_length]))
                    if prefix_length > 1
                    else matrices[0]
                )
                for prefix_length in range(1, 6)
            ]
        )
        actual = associative_scan(combine, elems, axis=1)
        np.testing.assert_array_equal(np.asarray(actual), expected.transpose(1, 0, 2))

    def test_dictionary_order_cannot_reassign_fields(self) -> None:
        elems = {"a": tk.array([1.0, 2.0, 3.0]), "b": tk.array([10.0, 20.0, 30.0])}

        def combine(
            left: dict[str, tk.array], right: dict[str, tk.array]
        ) -> dict[str, tk.array]:
            return {"b": left["b"] + right["b"], "a": left["a"] + right["a"]}

        for reverse in (False, True):
            result = associative_scan(combine, elems, reverse=reverse)
            for key in elems:
                self.assertTrue(
                    tk.array_equal(result[key], tk.cumsum(elems[key], reverse=reverse))
                )

    def test_container_types_and_literal_keys_survive(self) -> None:
        array = tk.array([1.0, 2.0, 3.0])
        for elems in ((array, array), Pair(array, array)):

            def combine(
                left: tuple[tk.array, tk.array], right: tuple[tk.array, tk.array]
            ) -> tuple[tk.array, tk.array]:
                self.assertIs(type(left), type(elems))
                self.assertIs(type(right), type(elems))
                values = (left[0] + right[0], left[1] + right[1])
                return Pair(*values) if isinstance(elems, Pair) else values

            result = associative_scan(combine, elems)
            self.assertIs(type(result), type(elems))
        elems: LiteralTree = {"a.b": array, "a": {"b": array}, "empty": ()}

        def combine(left: LiteralTree, right: LiteralTree) -> LiteralTree:
            return {
                "a.b": left["a.b"] + right["a.b"],
                "a": {"b": left["a"]["b"] + right["a"]["b"]},
                "empty": (),
            }

        result = associative_scan(combine, elems)
        self.assertEqual(result.keys(), elems.keys())
        self.assertEqual(result["empty"], ())
        self.assertTrue(tk.array_equal(result["a.b"], tk.cumsum(array)))

    def test_combine_must_preserve_structure_and_leaf_shapes(self) -> None:
        array = tk.ones((3,))
        with self.assertRaises(ValueError):
            associative_scan(lambda left, right: (left[0] + right[0],), (array, array))
        with self.assertRaises(ValueError):
            associative_scan(
                lambda left, right: {"wrong": left["a"] + right["a"]}, {"a": array}
            )
        with self.assertRaises(ValueError):
            associative_scan(lambda left, right: (left + right)[:0], array)


if __name__ == "__main__":
    unittest.main()
