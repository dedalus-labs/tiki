# Copyright © 2023 Apple Inc.

import os
import platform
import sys
import unittest
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import tiki as tk


class TIKITestRunner(unittest.TestProgram):
    def __init__(self, *args, **kwargs):
        # Do not exit in runTests
        kwargs["exit"] = False
        super().__init__(*args, **kwargs)

    def runTests(self):
        super().runTests()
        tk.clear_streams()
        sys.exit(0 if self.result.wasSuccessful() else 1)


class TIKITestCase(unittest.TestCase):
    @property
    def is_apple_silicon(self):
        return platform.machine() == "arm64" and platform.system() == "Darwin"

    def setUp(self):
        self.default = tk.default_device()
        device = os.getenv("DEVICE", None)
        if device is not None:
            device = getattr(tk, device)
            tk.set_default_device(device)

    def tearDown(self):
        tk.set_default_device(self.default)

    # Note if a tuple is passed into args, it will be considered a shape request and convert to a tk.random.normal with the shape matching the tuple
    def assertCmpNumpy(
        self,
        args: List[Union[Tuple[int], Any]],
        mx_fn: Callable[..., tk.array],
        np_fn: Callable[..., np.array],
        atol=1e-2,
        rtol=1e-2,
        dtype=tk.float32,
        **kwargs,
    ):
        assert dtype != tk.bfloat16, "numpy does not support bfloat16"
        args = [
            tk.random.normal(s, dtype=dtype) if isinstance(s, Tuple) else s
            for s in args
        ]
        mx_res = mx_fn(*args, **kwargs)
        np_res = np_fn(
            *[np.array(a) if isinstance(a, tk.array) else a for a in args], **kwargs
        )
        return self.assertEqualArray(mx_res, tk.array(np_res), atol=atol, rtol=rtol)

    def assertEqualArray(
        self,
        mx_res: tk.array,
        expected: tk.array,
        atol=1e-2,
        rtol=1e-2,
    ):
        self.assertEqual(
            tuple(mx_res.shape),
            tuple(expected.shape),
            msg=f"shape mismatch expected={expected.shape} got={mx_res.shape}",
        )
        self.assertEqual(
            mx_res.dtype,
            expected.dtype,
            msg=f"dtype mismatch expected={expected.dtype} got={mx_res.dtype}",
        )
        if not isinstance(mx_res, tk.array) and not isinstance(expected, tk.array):
            np.testing.assert_allclose(mx_res, expected, rtol=rtol, atol=atol)
            return
        elif not isinstance(mx_res, tk.array):
            mx_res = tk.array(mx_res)
        elif not isinstance(expected, tk.array):
            expected = tk.array(expected)
        self.assertTrue(tk.allclose(mx_res, expected, rtol=rtol, atol=atol))
