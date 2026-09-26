# Copyright © 2026 Dedalus Labs, Inc.

"""The published layout examples execute as part of the Python test suite."""

import doctest
import unittest
from pathlib import Path


def load_tests(
    loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None
) -> unittest.TestSuite:
    source = Path(__file__).resolve().parents[2] / "docs" / "src"
    parser = doctest.DocTestParser()
    for page in ("usage/layouts.rst", "examples/layouts.rst"):
        path = source / page
        test = parser.get_doctest(path.read_text(), {}, page, str(path), 0)
        if not test.examples:
            raise ValueError(f"{page} must contain executable examples")
        tests.addTest(doctest.DocTestCase(test))
    return tests


if __name__ == "__main__":
    unittest.main()
