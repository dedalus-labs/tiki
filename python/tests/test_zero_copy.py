# Copyright © 2024 Apple Inc.

import faulthandler
import gc
import threading
import unittest

import numpy as np
import tiki as tk
import tiki_tests


class TestZeroCopy(tiki_tests.TIKITestCase):
    """Tests for zero-copy CPU import: tk.asarray(host_buffer, copy=False).

    On unified memory (Metal) a page-aligned CPU buffer is adopted instead of
    copied. On backends without Metal, or for a non-page-aligned buffer or a
    dtype conversion, copy=False raises.
    """

    def test_default_shares_or_copies(self):
        # With copy=None Tiki adopts the buffer when it can and copies otherwise,
        # so either way the values must match the source at import time.
        a = np.arange(1_000_000, dtype=np.int32)
        x = tk.asarray(a)
        tk.eval(x)
        self.assertTrue(np.array_equal(np.array(x), a))

    def test_copy_true_copies(self):
        a = np.arange(1_000_000, dtype=np.int32)
        x = tk.asarray(a, copy=True)
        a[0] = 12345
        tk.eval(x)
        self.assertNotEqual(int(x[0]), 12345)

    def test_copy_false(self):
        a = np.arange(1_000_000, dtype=np.int32)
        if not tk.metal.is_available():
            with self.assertRaises(Exception):
                tk.asarray(a, copy=False)
            return
        if a.ctypes.data % 16384 != 0:
            self.skipTest("source buffer not page-aligned; adopt path not taken")
        x = tk.asarray(a, copy=False)
        self.assertTrue(np.array_equal(np.array(x), a))
        # Zero-copy adoption: a mutation of the source is visible in the array.
        a[1] = 999
        tk.eval(x)
        self.assertEqual(int(x[1]), 999)

    def test_copy_false_dtype_conversion_raises(self):
        a = np.arange(16, dtype=np.float64)
        with self.assertRaises(Exception):
            tk.asarray(a, dtype=tk.float32, copy=False)

    def test_source_lifetime(self):
        if not tk.metal.is_available():
            self.skipTest("copy=False requires Metal")

        def make():
            a = np.arange(1_000_000, dtype=np.float32) + 0.5
            if a.ctypes.data % 16384 != 0:
                return None
            return tk.asarray(a, copy=False)

        x = make()
        if x is None:
            self.skipTest("source buffer not page-aligned")
        gc.collect()
        tk.eval(x + 1)
        self.assertAlmostEqual(float(x[10]), 10.5, places=5)

    def test_adopt_in_loop_not_recycled(self):
        # Regression: an adopted buffer must be released (not recycled into the
        # allocator's reuse pool) when freed. Otherwise, over many iterations the
        # pool hands a caller-owned buffer to an unrelated array and corrupts /
        # crashes. Adopt fresh buffers in a loop and compute after each.
        if not tk.metal.is_available():
            self.skipTest("copy=False requires Metal")
        w = tk.random.normal((64, 64))
        for i in range(200):
            a = np.random.rand(256, 64).astype(np.float32)
            x = tk.asarray(a, copy=False)  # adopt; x (and a) freed next iteration
            r = tk.sum(x @ w)
            tk.eval(r)
        self.assertTrue(True)  # reaching here without crashing is the assertion

    def _adopted_source(self, n):
        # A square source buffer that tk.asarray can adopt, else a skip.
        if not tk.metal.is_available():
            self.skipTest("copy=False requires Metal")
        a = np.zeros((n, n), dtype=np.float32)
        if a.ctypes.data % 16384 != 0:
            self.skipTest("source buffer not page-aligned; adopt path not taken")
        return a

    @staticmethod
    def _submit_work(a):
        # The adopted array is an input of the first matmul and is dropped here,
        # so only the completion handler of the command buffer that is in flight
        # keeps its Python owner. The caller must keep the output until the end.
        n = a.shape[0]
        w = tk.ones((n, n))
        y = tk.asarray(a, copy=False) @ w
        for _ in range(4):
            y = y @ w
        tk.async_eval(y)
        return y

    # A stream callback can free an adopted buffer, which takes the GIL. A call
    # that waits for a stream must release the GIL, else the two deadlock.
    # faulthandler reports such a deadlock, its timer is a C thread. A watchdog
    # in Python would never run.
    def test_synchronize_releases_gil(self):
        a = self._adopted_source(1024)
        faulthandler.dump_traceback_later(30, exit=True)
        try:
            for _ in range(4):
                y = self._submit_work(a)
                tk.synchronize()
                del y
        finally:
            faulthandler.cancel_dump_traceback_later()

    def test_clear_streams_releases_gil(self):
        a = self._adopted_source(1024)
        errors = []

        # clear_streams destroys the streams of the calling thread, so it runs
        # in a thread that ends right after.
        def worker():
            try:
                y = self._submit_work(a)
                tk.clear_streams()
                del y
            except Exception as e:
                errors.append(e)

        faulthandler.dump_traceback_later(30, exit=True)
        try:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
        finally:
            faulthandler.cancel_dump_traceback_later()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
