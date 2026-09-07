# Copyright © 2023 Apple Inc.

import unittest

import tiki as tk
import tiki.nn as nn
import tiki.utils
import tiki_tests


class TestTreeUtils(tiki_tests.TIKITestCase):
    def test_tree_map(self):
        tree = {"a": 0, "b": 1, "c": 2}
        tree = tiki.utils.tree_map(lambda x: x + 1, tree)

        expected_tree = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(tree, expected_tree)

    def test_tree_flatten(self):
        tree = [{"a": 1, "b": 2}, "c"]
        vals = (1, 2, "c")
        flat_tree = tiki.utils.tree_flatten(tree)
        self.assertEqual(list(zip(*flat_tree))[1], vals)
        self.assertEqual(tiki.utils.tree_unflatten(flat_tree), tree)

    def test_merge(self):
        t1 = {"a": 0}
        t2 = {"b": 1}
        t = tiki.utils.tree_merge(t1, t2)
        self.assertEqual({"a": 0, "b": 1}, t)
        with self.assertRaises(ValueError):
            tiki.utils.tree_merge(t1, t1)
        with self.assertRaises(ValueError):
            tiki.utils.tree_merge(t, t1)

        mod1 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        mod2 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        mod = nn.Sequential(mod1, mod2)

        params1 = {"layers": [mod1.parameters()]}
        params2 = {"layers": [None, mod2.parameters()]}
        params = tiki.utils.tree_merge(params1, params2)
        for (k1, v1), (k2, v2) in zip(
            tiki.utils.tree_flatten(params), tiki.utils.tree_flatten(mod.parameters())
        ):
            self.assertEqual(k1, k2)
            self.assertTrue(tk.array_equal(v1, v2))

    def test_empty_subtree_merge(self):
        # make sure tiki pytrees treat empty dict {} as an empty node
        self.assertEqual([], tiki.utils.tree_flatten({"a": {}}))

        # empty dict merging
        self.assertEqual({}, tiki.utils.tree_merge({}, {}))
        self.assertEqual(
            [{"a": 1, "b": 2}, {}], tiki.utils.tree_merge([{"a": 1}, {}], [{"b": 2}, {}])
        )
        self.assertEqual({"a": {}}, tiki.utils.tree_merge({"a": {}}, {"a": {}}))
        self.assertEqual(
            {"a": 1, "b": {}, "c": 2},
            tiki.utils.tree_merge(
                {"a": 1, "b": {}, "c": {}}, {"a": {}, "b": {}, "c": 2}
            ),
        )

        # empty list merging
        self.assertEqual({"a": []}, tiki.utils.tree_merge({"a": []}, {"a": []}))

        # empty tuple merging
        self.assertEqual({"a": ()}, tiki.utils.tree_merge({"a": ()}, {"a": ()}))

        # merging different empty structures
        with self.assertRaises(ValueError):
            tiki.utils.tree_merge({}, [])

        def merge_called(a, b):
            raise AssertionError("merge_fn called on empty subtrees")

        self.assertEqual(
            {"a": {}}, tiki.utils.tree_merge({"a": {}}, {"a": {}}, merge_called)
        )

    def test_supported_trees(self):

        from typing import NamedTuple

        class Vector(tuple):
            pass

        class Params(NamedTuple):
            m: tk.array
            b: tk.array

        list1 = [tk.array([0, 1]), tk.array(2)]
        tuple1 = (tk.array([0, 1]), tk.array(2))
        vector1 = Vector([tk.array([0, 1]), tk.array(2)])
        params1 = Params(m=tk.array([0, 1]), b=tk.array(2))
        dict1 = {"m": tk.array([0, 1]), "b": tk.array(2)}

        add_one = lambda x: x + 1

        list2 = tiki.utils.tree_map(add_one, list1)
        tuple2 = tiki.utils.tree_map(add_one, tuple1)
        vector2 = tiki.utils.tree_map(add_one, vector1)
        params2 = tiki.utils.tree_map(add_one, params1)
        dict2 = tiki.utils.tree_map(add_one, dict1)

        self.assertTrue(isinstance(list2, list))
        self.assertTrue(tk.array_equal(list2[0], tk.array([1, 2])))
        self.assertTrue(tk.array_equal(list2[1], tk.array(3)))

        self.assertTrue(isinstance(tuple2, tuple))
        self.assertTrue(tk.array_equal(tuple2[0], tk.array([1, 2])))
        self.assertTrue(tk.array_equal(tuple2[1], tk.array(3)))

        self.assertTrue(isinstance(vector2, Vector))
        self.assertTrue(tk.array_equal(vector2[0], tk.array([1, 2])))
        self.assertTrue(tk.array_equal(vector2[1], tk.array(3)))

        self.assertTrue(isinstance(dict2, dict))
        self.assertTrue(tk.array_equal(dict2["m"], tk.array([1, 2])))
        self.assertTrue(tk.array_equal(dict2["b"], tk.array(3)))

        self.assertTrue(isinstance(params2, Params))
        self.assertTrue(tk.array_equal(params2.m, tk.array([1, 2])))
        self.assertTrue(tk.array_equal(params2.b, tk.array(3)))

        paths = []
        params3 = tiki.utils.tree_map_with_path(
            lambda path, x: paths.append(path) or x + 1, params1
        )

        self.assertEqual(paths, ["0", "1"])
        self.assertTrue(isinstance(params3, Params))
        self.assertTrue(tk.array_equal(params3.m, tk.array([1, 2])))
        self.assertTrue(tk.array_equal(params3.b, tk.array(3)))

        paths = []
        params4 = tiki.utils.tree_map_with_path(
            lambda path, x, y: paths.append(path) or x + y, params1, params3
        )

        self.assertEqual(paths, ["0", "1"])
        self.assertTrue(isinstance(params4, Params))
        self.assertTrue(tk.array_equal(params4.m, tk.array([1, 3])))
        self.assertTrue(tk.array_equal(params4.b, tk.array(5)))

        params5 = tiki.utils.tree_merge(params1, params1, lambda a, b: a + b)
        self.assertTrue(isinstance(params5, Params))
        self.assertTrue(tk.array_equal(params5.m, tk.array([0, 2])))
        self.assertTrue(tk.array_equal(params5.b, tk.array(4)))

        vector3 = tiki.utils.tree_merge(vector1, vector1, lambda a, b: a + b)
        self.assertTrue(isinstance(vector3, Vector))
        self.assertTrue(tk.array_equal(vector3[0], tk.array([0, 2])))
        self.assertTrue(tk.array_equal(vector3[1], tk.array(4)))

    def test_tree_unflatten_integer_key_collision(self):
        # Non-canonical integer-like keys (e.g. "01") must not silently
        # collide with "1" and shift later values. Fall back to dict tree.
        tree = tiki.utils.tree_unflatten([("01", "a"), ("1", "b"), ("2", "c")])
        self.assertIsInstance(tree, dict)
        self.assertEqual(tree["01"], "a")
        self.assertEqual(tree["1"], "b")
        self.assertEqual(tree["2"], "c")

        # Canonical list keys still unflatten as a list
        self.assertEqual(
            tiki.utils.tree_unflatten([("0", "a"), ("1", "b"), ("2", "c")]),
            ["a", "b", "c"],
        )


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
