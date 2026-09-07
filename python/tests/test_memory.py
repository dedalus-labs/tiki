# Copyright © 2023-2024 Apple Inc.

import unittest

import tiki as tk
import tiki_tests


class TestMemory(tiki_tests.TIKITestCase):
    def test_array_buffer_size(self):
        a = tk.array([1.0, 2.0, 3.0, 4.0])
        view = a[:1]
        lazy = a + 1

        with self.assertRaises(ValueError):
            tk.get_array_buffer_size({"lazy": lazy})

        tk.eval(view)
        a_size = tk.get_array_buffer_size(a)
        self.assertGreaterEqual(a_size, a.nbytes)
        self.assertLess(view.nbytes, a.nbytes)
        self.assertEqual(tk.get_array_buffer_size(), 0)
        self.assertEqual(tk.get_array_buffer_size(view), a_size)
        self.assertEqual(
            tk.get_array_buffer_size({"a": a, "nested": (a, None)}), a_size
        )
        self.assertEqual(tk.get_array_buffer_size(tk.array([])), 0)

        b = tk.array([5.0, 6.0])
        self.assertEqual(
            tk.get_array_buffer_size(a, b),
            a_size + tk.get_array_buffer_size(b),
        )

    def test_memory_info(self):
        old_limit = tk.set_cache_limit(0)

        a = tk.zeros((4096,))
        tk.eval(a)
        del a
        self.assertEqual(tk.get_cache_memory(), 0)
        self.assertEqual(tk.set_cache_limit(old_limit), 0)
        self.assertEqual(tk.set_cache_limit(old_limit), old_limit)

        old_limit = tk.set_memory_limit(10)
        self.assertEqual(tk.set_memory_limit(old_limit), 10)
        self.assertEqual(tk.set_memory_limit(old_limit), old_limit)

        # Query active and peak memory
        a = tk.zeros((4096,))
        tk.eval(a)
        tk.synchronize()
        active_mem = tk.get_active_memory()
        self.assertTrue(active_mem >= 4096 * 4)

        b = tk.zeros((4096,))
        tk.eval(b)
        del b
        tk.synchronize()

        new_active_mem = tk.get_active_memory()
        self.assertEqual(new_active_mem, active_mem)
        peak_mem = tk.get_peak_memory()
        self.assertTrue(peak_mem >= 4096 * 8)

        if tk.metal.is_available():
            cache_mem = tk.get_cache_memory()
            self.assertTrue(cache_mem >= 4096 * 4)

        tk.clear_cache()
        self.assertEqual(tk.get_cache_memory(), 0)

        tk.reset_peak_memory()
        self.assertEqual(tk.get_peak_memory(), 0)

    @unittest.skipIf(not tk.metal.is_available(), "Metal is not available")
    def test_wired_memory(self):
        old_limit = tk.set_wired_limit(1000)
        old_limit = tk.set_wired_limit(0)
        self.assertEqual(old_limit, 1000)

        max_size = tk.device_info(tk.gpu)["max_recommended_working_set_size"]
        with self.assertRaises(ValueError):
            tk.set_wired_limit(max_size + 10)

    def test_active_memory_count(self):
        tk.synchronize()
        tk.clear_cache()
        init_mem = tk.get_active_memory()
        a = tk.zeros((128, 128))
        tk.eval(a)
        tk.synchronize()
        del a
        a = tk.zeros((90, 128))
        tk.eval(a)
        tk.synchronize()
        del a
        self.assertEqual(init_mem, tk.get_active_memory())


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
