# Copyright © 2023 Apple Inc.

import unittest
from functools import partial

import tiki as tk
import tiki_tests


class TestEval(tiki_tests.TIKITestCase):
    def test_eval(self):
        arrs = [tk.ones((2, 2)) for _ in range(4)]
        tk.eval(*arrs)
        for x in arrs:
            self.assertEqual(x.tolist(), [[1, 1], [1, 1]])

    def test_retain_graph(self):
        def fun(x):
            y = 3 * x
            tk.eval(y)
            return 2 * y

        dfun_dx = tk.grad(fun)
        y = dfun_dx(tk.array(1.0))
        self.assertEqual(y.item(), 6.0)

    def test_eval_mixed(self):
        x = tk.array(1) + 1 + 1
        y = 0
        z = "hello"
        state = [x, y, z]
        tk.eval(state)
        self.assertEqual(x.item(), 3)

    def test_async_eval(self):
        x = tk.array(1) + tk.array(1) + tk.array(1)
        tk.async_eval(x)
        self.assertEqual(x.item(), 3)

        # It should be safe to call eval on the array which has been async
        # eval'ed
        x = tk.array(1) + tk.array(1) + tk.array(1)
        self.assertEqual(x.item(), 3)

        x = tk.array([1, 2, 3])
        y = 2 * x
        tk.async_eval(y)
        z = 2 * y
        tk.async_eval(z)
        self.assertTrue(tk.array_equal(y, tk.array([2, 4, 6])))
        self.assertTrue(tk.array_equal(z, tk.array([4, 8, 12])))

    def test_async_eval_twice(self):
        for _ in range(1000):
            x = tk.array(1) + tk.array(1) + tk.array(1)
            tk.async_eval(x)
            y = x + 1
            tk.async_eval(y)
            self.assertEqual(x.item(), 3)
            self.assertEqual(y.item(), 4)

    def test_async_eval_in_trace(self):
        def fun(x):
            y = x + 1.0
            tk.async_eval(y)
            return tk.exp(y)

        # Raises
        with self.assertRaises(ValueError):
            tk.grad(fun)(tk.array(1.0))

        # Also raises
        with self.assertRaises(ValueError):
            tk.vmap(fun)(tk.ones((2, 2)))

    def test_async_eval_into_eval(self):
        x = tk.array(1)
        y = x + 1
        tk.async_eval(y)
        a = y - 10
        b = tk.abs(a)
        self.assertEqual(b.item(), 8)

    def test_async_eval_into_eval_diff_stream(self):
        s = tk.new_stream(tk.cpu)
        x = tk.array(0)
        y = x - 5
        tk.async_eval(y)
        z = tk.abs(y, stream=s)
        self.assertEqual(z.item(), 5)

    def test_eval_slow_fast_multi_stream(self):
        x = tk.ones((8000,))
        y = tk.abs(tk.array(-1.0))
        for _ in range(20):
            x = x + tk.array(1.0)
        z = tk.add(x, y, stream=tk.cpu)
        self.assertTrue(tk.allclose(z, tk.full((8000,), 22.0)))

        # Switch eval order
        x = tk.ones((8000,))
        y = tk.abs(tk.array(-1.0))
        for _ in range(20):
            x = x + tk.array(1.0)
        z = tk.add(y, x, stream=tk.cpu)
        self.assertTrue(tk.allclose(z, tk.full((8000,), 22.0)))

    def test_multi_output_eval_during_transform(self):
        x = tk.random.uniform(shape=(1024,))
        y = tk.ones((1024,))
        tk.eval(x, y)

        def fn(x):
            a, b = tk.divmod(x, x)
            tk.eval(a)
            return a

        out = tk.vjp(fn, (x,), (y,))
        out = tk.vjp(fn, (x,), (y,))
        peak_mem = tk.get_peak_memory()
        out = tk.vjp(fn, (x,), (y,))
        self.assertEqual(peak_mem, tk.get_peak_memory())

    def test_async_eval_with_multiple_streams(self):
        x = tk.array([1.0])
        y = tk.array([1.0])
        a = tk.array([1.0])
        b = tk.array([1.0])

        d = tk.default_device()
        s2 = tk.new_stream(d)

        for _ in range(50):
            for _ in range(20):
                x = x + y
            tk.async_eval(x)
            tk.eval(a + b)

    def test_donation_for_noops(self):
        def fun(x):
            s = x.shape
            for _ in range(10):
                x = tk.abs(x)
                x = tk.reshape(x, (-1,))
                x = x.T.T
                x = tk.stop_gradient(x)
                x = tk.abs(x)
            return x

        x = tk.zeros((4096, 4096))
        tk.eval(x)
        pre = tk.get_peak_memory()
        out = fun(x)
        del x
        tk.eval(out)
        post = tk.get_peak_memory()
        self.assertEqual(pre, post)

        def fun(x):
            for _ in range(10):
                x = tk.abs(x)
                x = x[:-1]
                x = tk.abs(x)
            return x

        x = tk.zeros((4096 * 4096,))
        tk.eval(x)
        pre = tk.get_peak_memory()
        out = fun(x)
        del x
        tk.eval(out)
        post = tk.get_peak_memory()
        self.assertEqual(pre, post)

    @unittest.skipIf(not tk.is_available(tk.gpu), "GPU is not available")
    def test_multistream_deadlock(self):
        s1 = tk.default_stream(tk.gpu)
        s2 = tk.new_stream(tk.gpu)

        x = tk.array(1.0)
        x = tk.abs(x, stream=s1)
        for _ in range(1000):
            x = tk.abs(x, stream=s2)
        tk.eval(x)

        s1 = tk.default_stream(tk.gpu)
        s2 = tk.new_stream(tk.gpu)
        old_limit = tk.set_memory_limit(1000)

        x = tk.ones((512, 512), stream=s2)
        for _ in range(80):
            x = tk.abs(x, stream=s1)
        y = tk.abs(x, stream=s2)
        z = tk.abs(y, stream=s2)
        tk.eval(z)
        tk.set_memory_limit(old_limit)

    @unittest.skipIf(not tk.metal.is_available(), "Metal is not available")
    def test_eval_exception_does_not_corrupt_state(self):
        # An exception thrown from inside a primitive's eval (here a Metal
        # compile error raised lazily at eval time) must not corrupt arrays
        # evaluated earlier in the same batch: they are already marked
        # evaluated, so their pending command buffers must still be
        # committed before the exception propagates.
        a = tk.full((1024,), 3.0)
        b = a * 2.0  # encoded in the same eval batch as the failing kernel

        kernel = tk.fast.metal_kernel(
            name="test_eval_exception_bad_kernel",
            input_names=["inp"],
            output_names=["out"],
            source="this is not metal code {",
        )
        with self.assertRaises(Exception):
            (y,) = kernel(
                inputs=[b],
                output_shapes=[b.shape],
                output_dtypes=[b.dtype],
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
            )
            tk.eval(y)

        self.assertTrue(tk.all(b == 6.0).item())

        # Fresh computations after the failure stay correct.
        x = tk.full((512,), 2.0)
        self.assertEqual((x + 1.0).sum().item(), 512.0 * 3.0)

    @unittest.skipIf(
        tk.cuda.is_available(), "CUDA backend waits cpu stream synchronously"
    )
    def test_async_eval_error_in_synchronize(self):
        a = tk.linalg.inv(tk.array([[1.0, 2.0], [2.0, 4.0]]), stream=tk.cpu)
        tk.async_eval(a)
        with self.assertRaises(RuntimeError):
            tk.synchronize(tk.cpu)


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
