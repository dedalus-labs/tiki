# Copyright © 2023 Apple Inc.

import io
import unittest

import tiki as tk
import tiki_tests


class TestGraph(tiki_tests.TIKITestCase):
    def test_to_dot(self):
        # Simply test that a few cases run.
        # Nothing too specific about the graph format
        # for now to keep it flexible
        a = tk.array(1.0)
        f = io.StringIO()
        tk.export_to_dot(f, a)
        f.seek(0)
        self.assertTrue(len(f.read()) > 0)

        b = tk.array(2.0)
        c = a + b
        f = io.StringIO()
        tk.export_to_dot(f, c)
        f.seek(0)
        self.assertTrue(len(f.read()) > 0)

        # Multi output case
        c = tk.divmod(a, b)
        f = io.StringIO()
        tk.export_to_dot(f, *c)
        f.seek(0)
        self.assertTrue(len(f.read()) > 0)


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
