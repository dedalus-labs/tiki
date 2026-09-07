# Copyright © 2026 Apple Inc.

import threading
import unittest

import tiki as tk
import tiki_tests


class TestThreads(tiki_tests.TIKITestCase):
    def test_threadlocal_stream(self):
        raised = [False, False]
        test_stream = tk.new_stream(tk.default_device())

        def test_failure(i):
            with self.assertRaises(RuntimeError):
                with tk.stream(test_stream):
                    x = tk.arange(10)
                    tk.eval(2 * x)
            tk.clear_streams()
            raised[i] = True

        t1 = threading.Thread(target=test_failure, args=(0,))
        t2 = threading.Thread(target=test_failure, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertTrue(all(raised))

        raised = [True, True]
        test_stream = tk.new_thread_local_stream(tk.default_device())

        def test_success(i):
            with tk.stream(test_stream):
                x = tk.arange(10)
                tk.eval(2 * x)
                self.assertEqual(x.tolist(), list(range(10)))
            tk.clear_streams()
            raised[i] = False

        t1 = threading.Thread(target=test_success, args=(0,))
        t2 = threading.Thread(target=test_success, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertFalse(any(raised))

    def test_concurrent_transforms(self):
        # Running function transformations from multiple threads concurrently
        # must not crash or corrupt results. The tracing state used to mark
        # "we are inside a transformation" is per-thread, so independent traces
        # on different threads do not interfere.
        x = tk.array([1.0, 2.0, 3.0])
        n_iters = 2000

        # Single-threaded references.
        expected_grad = tk.grad(lambda a: (a * a).sum())(x).tolist()
        expected_compile = tk.compile(lambda a: a * a + 1)(x).tolist()
        xb = tk.broadcast_to(x, (8, 3))
        expected_vmap = tk.vmap(lambda a: a * a)(xb).tolist()

        errors = []

        def grad_worker():
            try:
                g = tk.grad(lambda a: (a * a).sum())
                for _ in range(n_iters):
                    r = g(x)
                    tk.eval(r)
                    if r.tolist() != expected_grad:
                        errors.append(("grad", r.tolist()))
                        return
            except Exception as e:
                errors.append(("grad", repr(e)))

        def compile_worker():
            try:
                f = tk.compile(lambda a: a * a + 1)
                for _ in range(n_iters):
                    r = f(x)
                    tk.eval(r)
                    if r.tolist() != expected_compile:
                        errors.append(("compile", r.tolist()))
                        return
            except Exception as e:
                errors.append(("compile", repr(e)))

        def vmap_worker():
            try:
                h = tk.vmap(lambda a: a * a)
                for _ in range(n_iters):
                    r = h(xb)
                    tk.eval(r)
                    if r.tolist() != expected_vmap:
                        errors.append(("vmap", r.tolist()))
                        return
            except Exception as e:
                errors.append(("vmap", repr(e)))

        workers = [grad_worker, compile_worker, vmap_worker]
        threads = [threading.Thread(target=workers[i % 3]) for i in range(9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
