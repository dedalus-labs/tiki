# Copyright © 2023-2024 Apple Inc.

import gc
import inspect
import io
import math
import threading
from functools import partial, wraps
from io import StringIO

import numpy as np
import tiki as tk
import tiki_tests


class TestCompile(tiki_tests.TIKITestCase):
    def test_simple_compile(self):
        def fun(x, y):
            return x + y

        compiled_fn = tk.compile(fun)
        compiled_fn = tk.compile(fun)
        x = tk.array(1.0)
        y = tk.array(1.0)
        out = compiled_fn(x, y)
        self.assertEqual(out.item(), 2.0)

        # Try again
        out = compiled_fn(x, y)
        self.assertEqual(out.item(), 2.0)

        # Change sizes
        x = tk.array([1.0, 2.0])
        out = compiled_fn(x, y)
        self.assertTrue(tk.array_equal(out, tk.array([2.0, 3.0])))

        y = tk.array([1.0, 2.0])
        out = compiled_fn(x, y)
        self.assertTrue(tk.array_equal(out, tk.array([2.0, 4.0])))

        # Change types
        x = tk.array([1, 2], tk.int32)
        y = tk.array([1, 2], tk.int32)
        out = compiled_fn(x, y)
        self.assertEqual(out.dtype, tk.int32)
        self.assertTrue(tk.array_equal(out, tk.array([2, 4])))

    def test_compile_nonfinite_constants(self):
        # Regression test: a non-finite scalar constant (NaN / infinity) baked
        # into a fused compiled kernel used to stream a bare token (e.g. `nan`)
        # into the generated kernel source, which is not a valid identifier and
        # broke compilation (notably on the Metal backend).
        x = tk.array([1.0, -1.0])

        for dtype in (tk.float32, tk.float16, tk.bfloat16):
            xd = x.astype(dtype)

            nan_fn = tk.compile(
                lambda a: tk.where(a > 0, a, tk.array(float("nan"), dtype=a.dtype))
            )
            out = nan_fn(xd)
            tk.eval(out)
            self.assertEqual(out[0].item(), 1.0)
            self.assertTrue(math.isnan(out[1].item()))

            neg_inf_fn = tk.compile(
                lambda a: tk.where(a > 0, a, tk.array(float("-inf"), dtype=a.dtype))
            )
            out = neg_inf_fn(xd)
            tk.eval(out)
            self.assertEqual(out[0].item(), 1.0)
            self.assertEqual(out[1].item(), float("-inf"))

    def test_compile_tuple_output_in_thread(self):
        @tk.compile
        def fun(x):
            return x + 1, x * 2

        results = []
        errors = []

        def worker():
            try:
                x = tk.array([1.0])
                y, z = fun(x)
                tk.eval(y, z)
                results.append((y.item(), z.item()))
            except Exception as e:
                errors.append(e)
            tk.clear_streams()

        for _ in range(3):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            gc.collect()

        if errors:
            raise errors[0]
        self.assertEqual(results, [(2.0, 2.0)] * 3)

    def test_compile_release_on_another_thread(self):
        # A function traced on one thread but released on another must still
        # drop its cache entry, otherwise a later compile of the same id gets
        # handed the dead function's tape instead of being traced again.
        traces = []

        def fun(x):
            traces.append(1)
            return x + 1

        holder = {}
        traced = threading.Event()
        released = threading.Event()
        errors = []

        def worker():
            try:
                holder["fn"] = tk.compile(fun)
                tk.eval(holder["fn"](tk.array([1.0])))
                traced.set()
                self.assertTrue(released.wait(10))
                # The same callable, so the same id.
                fn = tk.compile(fun)
                tk.eval(fn(tk.array([1.0])))
            except Exception as e:
                errors.append(e)
            finally:
                traced.set()
            tk.clear_streams()

        # The tracing thread has to outlive the release, on exit it would tear
        # down its cache anyway.
        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(traced.wait(10))
        holder.clear()
        gc.collect()
        released.set()
        thread.join()

        if errors:
            raise errors[0]
        self.assertEqual(len(traces), 2)

    def test_compile_grad(self):
        def loss_fn(x):
            return tk.exp(x).sum()

        grad_fn = tk.grad(loss_fn)

        x = tk.array([0.5, -0.5, 1.2])
        dfdx = grad_fn(x)
        compile_grad_fn = tk.compile(grad_fn)
        c_dfdx = grad_fn(x)

        self.assertTrue(tk.allclose(c_dfdx, dfdx))

        # Run it again without calling compile
        c_dfdx = compile_grad_fn(x)
        self.assertTrue(tk.allclose(c_dfdx, dfdx))

        # Run it again with calling compile
        c_dfdx = tk.compile(grad_fn)(x)
        self.assertTrue(tk.allclose(c_dfdx, dfdx))

        # Value and grad
        def loss_fn(x):
            return tk.exp(x).sum(), tk.sin(x)

        val_and_grad_fn = tk.value_and_grad(loss_fn)
        (loss, val), dfdx = val_and_grad_fn(x)
        (c_loss, c_val), c_dfdx = tk.compile(val_and_grad_fn)(x)

        self.assertTrue(tk.allclose(c_dfdx, dfdx))
        self.assertTrue(tk.allclose(c_loss, loss))
        self.assertTrue(tk.allclose(c_val, val))

    def test_compile_inputs_with_primitives(self):
        x = tk.array([1, 2, 3])
        y = tk.array([1, 2, 3])
        for _ in range(5):
            x = x + y
            y = y + 1

        def fun(x, y):
            return x * y

        out = fun(x, y)

        x = tk.array([1, 2, 3])
        y = tk.array([1, 2, 3])
        for _ in range(5):
            x = x + y
            y = y + 1

        c_out = tk.compile(fun)(x, y)
        self.assertTrue(tk.array_equal(out, c_out))

        # Try again
        c_out = tk.compile(fun)(x, y)
        self.assertTrue(tk.array_equal(out, c_out))

    def test_compile_with_closure(self):
        x = tk.array(1)

        def closure(y):
            return x + y

        compiled = tk.compile(closure)
        out = compiled(tk.array(1))
        self.assertEqual(out.item(), 2)

        # Try again
        out = compiled(tk.array(1))
        self.assertEqual(out.item(), 2)

        # Change the shape of the enclosed variable
        x = tk.array([1, 2])
        out = compiled(tk.array(1))

        # We still get the original input (closures are not updated)
        self.assertEqual(out.item(), 2)

        # Try with a tree of enclosed variables
        x = {"a": tk.array(1), "b": tk.array(2)}

        def closure(y):
            return x["a"] + y + x["b"]

        compiled = tk.compile(closure)
        out = compiled(tk.array(1))
        self.assertEqual(out.item(), 4)

        # Change the shape of one input
        x["a"] = tk.array([4, 5])
        out = compiled(tk.array(1))
        self.assertEqual(out.item(), 4)

        x["b"] = tk.array([-6, -8])
        out = compiled(tk.array(1))
        self.assertEqual(out.item(), 4)

        # Enclosed variable is not evaluated yet
        x = tk.array(1)
        x = x + x

        def closure(y):
            return x + y

        compiled = tk.compile(closure)
        out = compiled(tk.array(2))
        self.assertEqual(out.item(), 4)

        # And again
        out = compiled(tk.array(2))
        self.assertEqual(out.item(), 4)

    def test_function_creates_array(self):
        def fun(x):
            return x + tk.array(1)

        cfun = tk.compile(fun)
        out = cfun(tk.array(3))
        self.assertEqual(out.item(), 4)

        # And again
        out = cfun(tk.array(3))
        self.assertEqual(out.item(), 4)

    def test_enable_disable(self):
        def fun(x):
            y = x + 1
            z = x + 1
            return y + z

        def count_prims(outputs):
            buf = io.StringIO()
            tk.export_to_dot(buf, outputs)
            buf.seek(0)
            return len([l for l in buf.read().split() if "label" in l])

        x = tk.array(1.0)
        cfun = tk.compile(fun)
        n_compiled = count_prims(cfun(x))

        # Check disabled
        tk.disable_compile()
        n_uncompiled = count_prims(cfun(x))
        self.assertTrue(n_compiled < n_uncompiled)

        # Check renabled
        tk.enable_compile()
        n_enable_compiled = count_prims(cfun(x))
        self.assertEqual(n_compiled, n_enable_compiled)

    def test_compile_two_input_grad(self):
        def loss(w, x):
            y = x * w
            return (y * tk.exp(y)).sum()

        x = tk.array([1.0, 0.5, 2.0, -0.5])
        w = tk.array([-1.0, 0.3, 1.0, -0.9])

        expected_grad = tk.grad(loss)(w, x)
        compiled_grad = tk.compile(tk.grad(loss))(w, x)
        self.assertTrue(tk.allclose(expected_grad, compiled_grad))

    def test_vmap_compiled(self):
        def simple_unary(x):
            return -tk.exp(x)

        x = tk.array([[1.0, 2.0], [2.0, 3.0]])

        expected_out = tk.vmap(simple_unary)(x)
        out = tk.vmap(tk.compile(simple_unary))(x)
        self.assertTrue(tk.allclose(expected_out, out))

        def simple_binary(x, y):
            return tk.abs(tk.exp(x + y) + y)

        x = tk.array([[1.0, -3.0], [0.5, -0.5]])
        y = tk.array([[2.0, -1.0], [0.25, -0.25]])

        expected_out = tk.vmap(simple_binary)(x, y)
        out = tk.vmap(tk.compile(simple_binary))(x, y)
        self.assertTrue(tk.allclose(expected_out, out))

        expected_out = tk.vmap(simple_binary, in_axes=(0, 1))(x, y)
        out = tk.vmap(tk.compile(simple_binary), in_axes=(0, 1))(x, y)
        self.assertTrue(tk.allclose(expected_out, out))

        y = tk.array([0.25, -0.25])
        expected_out = tk.vmap(simple_binary, in_axes=(0, None))(x, y)
        out = tk.vmap(tk.compile(simple_binary), in_axes=(0, None))(x, y)
        self.assertTrue(tk.allclose(expected_out, out))

        def simple_unary_outer(x):
            x = tk.abs(x)

            @tk.compile
            def simple_unary_inner(z):
                return -tk.exp(x)

            return simple_unary_inner(x)

        expected_out = -tk.exp(tk.abs(x))
        out = tk.vmap(simple_unary_outer)(x)
        self.assertTrue(tk.allclose(expected_out, out))

    def test_vjp_vjp_compiled(self):
        def simple_unary(x):
            return -tk.exp(x)

        x = tk.array([[1.0, 2.0], [2.0, 3.0]])
        y = tk.array([[1.0, 1.0], [1.0, 1.0]])

        expected_out, expected_vjp_out = tk.vjp(simple_unary, (x,), (y,))
        out, vjp_out = tk.vjp(tk.compile(simple_unary), (x,), (y,))
        self.assertTrue(tk.allclose(expected_vjp_out[0], vjp_out[0]))
        self.assertTrue(tk.allclose(expected_out[0], out[0]))

        expected_out, expected_jvp_out = tk.jvp(simple_unary, (x,), (y,))
        out, jvp_out = tk.jvp(tk.compile(simple_unary), (x,), (y,))
        self.assertTrue(tk.allclose(expected_jvp_out[0], jvp_out[0]))
        self.assertTrue(tk.allclose(expected_out[0], out[0]))

        def simple_binary(x, y):
            return tk.abs(tk.exp(x + y) + y)

        x = tk.array([[1.0, -3.0], [0.5, -0.5]])
        y = tk.array([[2.0, -1.0], [0.25, -0.25]])
        cotans = tk.ones_like(x)

        expected_out, expected_vjp_out = tk.vjp(simple_binary, (x, y), (cotans,))
        out, vjp_out = tk.vjp(tk.compile(simple_binary), (x, y), (cotans,))
        self.assertTrue(tk.allclose(expected_out[0], out[0]))
        self.assertTrue(tk.allclose(expected_vjp_out[0], vjp_out[0]))
        self.assertTrue(tk.allclose(expected_vjp_out[1], vjp_out[1]))

        tans = (tk.ones_like(x), tk.ones_like(y))
        expected_out, expected_jvp_out = tk.jvp(simple_binary, (x, y), tans)
        out, jvp_out = tk.jvp(tk.compile(simple_binary), (x, y), tans)
        self.assertTrue(tk.allclose(expected_jvp_out[0], jvp_out[0]))
        self.assertTrue(tk.allclose(expected_out[0], out[0]))

    def test_transform_over_eval_compiled(self):
        def outer(x):
            y = tk.exp(tk.abs(x))
            tk.eval(y)
            return y.sum()

        x = tk.array([2.0, -1.0, 0.5])
        dfdx = tk.grad(outer)(x)

        @tk.compile
        def simple_unary(x):
            return tk.exp(tk.abs(x))

        def outer(x):
            y = simple_unary(x)
            tk.eval(y)
            return y.sum()

        cdfdx = tk.grad(outer)(x)
        self.assertTrue(tk.allclose(dfdx, cdfdx))

    def test_compile_capture(self):
        # Test update captured state outside compiled function
        state = {"y": tk.array(2)}

        @partial(tk.compile, inputs=state)
        def test_state(x):
            x = x + state["y"]
            return x

        test_state(tk.array(1))
        # Check the state is unchanged
        self.assertEqual(state["y"], 2)

        # Check the updated state is used
        state["y"] = tk.array(3)
        out = test_state(tk.array(1))
        self.assertEqual(out.item(), 4)

        # Capture list
        state = [tk.array(2)]

        @partial(tk.compile, inputs=state)
        def test_state(x):
            x = x + state[0]
            return x

        out = test_state(tk.array(1))
        self.assertEqual(out.item(), 3)
        state[0] = tk.array(3)
        out = test_state(tk.array(1))
        self.assertEqual(out.item(), 4)

        # Capture tuple of list
        state = ([tk.array(2)],)

        @partial(tk.compile, inputs=state)
        def test_state(x):
            x = x + state[0][0]
            return x

        out = test_state(tk.array(1))
        self.assertEqual(out.item(), 3)
        state[0][0] = tk.array(3)
        out = test_state(tk.array(1))
        self.assertEqual(out.item(), 4)

        # Test state updated inside compiled function
        state = {}

        @partial(tk.compile, outputs=state)
        def test_state(x):
            state["y"] = x + 3
            return tk.abs(x)

        test_state(tk.array(-1))
        self.assertEqual(state["y"].item(), 2)

        # Test state changed inside compiled function
        # triggers recompile
        state = {}

        @partial(tk.compile, inputs=state, outputs=state)
        def test_state(x):
            y = state.get("y", tk.array(0))
            state["y"] = x + y
            return x + 2 * y

        test_state(tk.array(1))
        self.assertEqual(state["y"].item(), 1)
        test_state(tk.array(1))
        self.assertEqual(state["y"].item(), 2)

    def test_compile_rng(self):
        @partial(tk.compile, inputs=tk.random.state, outputs=tk.random.state)
        def fun():
            return tk.random.uniform(shape=(10, 10))

        self.assertFalse(tk.allclose(fun(), fun(), 1e-2, 1e-2))

    def test_compile_rng_across_threads(self):
        # A function compiled with inputs/outputs=tk.random.state on one thread
        # must still use (and advance/seed) the calling thread's RNG state when
        # invoked from another thread, whether captured directly or nested.

        # The state sentinel is a single global object shared across threads.
        state_from_thread = {}

        def grab():
            state_from_thread["s"] = tk.random.state
            tk.clear_streams()

        t = threading.Thread(target=grab)
        t.start()
        t.join()
        self.assertIs(tk.random.state, state_from_thread["s"])

        direct = partial(tk.compile, inputs=tk.random.state, outputs=tk.random.state)(
            lambda: tk.random.uniform(shape=(10, 10))
        )

        nested_state = [{"unused": tk.array(0.0)}, tk.random.state]
        nested = partial(tk.compile, inputs=nested_state, outputs=nested_state)(
            lambda: tk.random.uniform(shape=(10, 10))
        )

        for fun in (direct, nested):
            results = {}

            def worker():
                with tk.stream(tk.cpu):
                    a = fun()
                    b = fun()
                    results["advances"] = not bool(tk.allclose(a, b, 1e-2, 1e-2).item())
                    tk.random.seed(42)
                    c = fun()
                    tk.random.seed(42)
                    d = fun()
                    results["seed_reproducible"] = bool(tk.allclose(c, d).item())
                    tk.random.seed(1234)
                    e = fun()
                    results["seed_changes"] = not bool(
                        tk.allclose(c, e, 1e-2, 1e-2).item()
                    )
                tk.clear_streams()

            t = threading.Thread(target=worker)
            t.start()
            t.join()

            self.assertTrue(results["advances"])
            self.assertTrue(results["seed_reproducible"])
            self.assertTrue(results["seed_changes"])

    def test_compile_state_capture_with_rng_updates_in_place(self):
        # Capturing tk.random.state alongside other state via outputs= must not
        # break in-place updates of the other captured containers.
        counter = {"v": tk.array(0.0)}
        state = [counter, tk.random.state]

        @partial(tk.compile, inputs=state, outputs=state)
        def step():
            counter["v"] = counter["v"] + 1.0
            return tk.random.uniform(shape=(2,))

        for _ in range(3):
            step()
        tk.eval(counter["v"])
        self.assertEqual(counter["v"].item(), 3.0)

    def test_compile_kwargs(self):
        @tk.compile
        def fun(x, y, z):
            return x + y + z

        x = tk.array(1)
        y = tk.array(2)
        z = tk.array(3)
        out = fun(x, y=y, z=z)
        self.assertEqual(out.item(), 6)

    def test_shapeless_compile(self):
        y = 1

        @partial(tk.compile, shapeless=True)
        def fun(x):
            return x + y

        x = tk.array([1, 2])
        self.assertTrue(tk.array_equal(fun(x), tk.array([2, 3])))

        # The function is not recompiled, so the change
        # to y should not be reflected in the output
        y = 2
        x = tk.array([1, 2, 3])
        self.assertTrue(tk.array_equal(fun(x), tk.array([2, 3, 4])))

        # Type change recompiles
        x = tk.array([1.0, 2.0, 3.0])
        self.assertTrue(tk.array_equal(fun(x), tk.array([3.0, 4.0, 5.0])))

        # Dim change recompiles
        x = tk.array([[1, 2, 3]])
        self.assertTrue(tk.array_equal(fun(x), tk.array([[3, 4, 5]])))

    def test_shapeless_compile_with_broadcasts(self):
        x = tk.ones((2, 2))
        y = tk.array([2, 2])

        def fun(x, y):
            return x * y

        cfun = tk.compile(fun, shapeless=True)
        self.assertTrue(tk.array_equal(cfun(x, y), fun(x, y)))
        self.assertTrue(tk.array_equal(cfun(y, x), fun(y, x)))
        y = tk.array([[3]])
        self.assertTrue(tk.array_equal(cfun(x, y), fun(x, y)))
        self.assertTrue(tk.array_equal(cfun(y, x), fun(y, x)))

    def test_shapeless_compile_with_reduction(self):
        # Test shapeless compile with a reduction
        z = 1

        @partial(tk.compile, shapeless=True)
        def fun(x, y):
            return x + y.sum(0, keepdims=True) + z

        x = tk.ones((2, 2), tk.int32)
        y = tk.ones((2, 2), tk.int32)
        self.assertTrue(tk.array_equal(fun(x, y), tk.full(shape=(2, 2), vals=4)))
        x = tk.ones((3, 3), tk.int32)
        y = tk.ones((3, 3), tk.int32)
        z = 2
        self.assertTrue(tk.array_equal(fun(x, y), tk.full(shape=(3, 3), vals=5)))

        x1 = tk.array([[1, 2], [3, 4], [5, 6]])
        x2 = tk.array([[1, 2]])

        def fun(x):
            return x * x.sum(-1, keepdims=True)

        cfun = tk.compile(fun, shapeless=True)
        tk.eval(cfun(x1))
        self.assertTrue(tk.array_equal(fun(x2), cfun(x2)))

        def fun(x):
            return x * x.sum(-1, keepdims=False)

        cfun = tk.compile(fun, shapeless=True)
        self.assertTrue(tk.array_equal(fun(x2), cfun(x2)))

    def test_shapeless_compile_unflatten(self):
        x = tk.zeros((1, 1, 4 * 32))

        def fun(x):
            return tk.unflatten(x, -1, (4, -1))

        self.assertEqual(tk.compile(fun, shapeless=True)(x).shape, (1, 1, 4, 32))

    def test_shapeless_compile_gather(self):
        x = tk.zeros((1, 1, 32))

        def fun(x):
            return x[:, -1, :]

        self.assertEqual(tk.compile(fun, shapeless=True)(x).shape, (1, 32))

    def test_shapeless_compile_full_like(self):
        x_shape = (1, 1, 32)
        x = tk.zeros((x_shape))

        def zeros_fun(x):
            return tk.zeros_like(x)

        def ones_fun(x):
            return tk.ones_like(x)

        compiled_zero_like = tk.compile(zeros_fun, shapeless=True)
        compiled_ones_like = tk.compile(ones_fun, shapeless=True)

        self.assertEqual(compiled_zero_like(x).shape, x_shape)
        self.assertEqual(compiled_ones_like(x).shape, x_shape)

        y_shape = (2, 2, 16)
        y = tk.zeros(y_shape)

        self.assertEqual(compiled_zero_like(y).shape, y_shape)
        self.assertEqual(compiled_ones_like(y).shape, y_shape)

    def test_shapeless_compile_gather_qmm(self):
        K, N, num_experts = 64, 32, 4

        w = tk.random.normal((num_experts, N, K))
        qw, s, b = tk.quantize(w)
        tk.eval(qw, s, b)

        idx = tk.array([0, 1, 2, 3])
        x4 = tk.ones((num_experts, 4, K))
        x8 = tk.ones((num_experts, 8, K))

        def fn(x):
            return tk.gather_qmm(
                x, qw, s, b, lhs_indices=idx, rhs_indices=idx, transpose=True
            )

        cfn = tk.compile(fn, shapeless=True)

        self.assertEqual(cfn(x4).shape, fn(x4).shape)
        self.assertEqual(cfn(x8).shape, fn(x8).shape)

    def test_shapeless_compile_gather_mm(self):
        K, N, num_experts = 64, 32, 4

        idx = tk.array([0, 1, 2, 3])
        b = tk.random.normal((num_experts, K, N))
        tk.eval(b)

        x4 = tk.ones((num_experts, 4, K))
        x8 = tk.ones((num_experts, 8, K))

        def fn(x):
            return tk.gather_mm(x, b, lhs_indices=idx, rhs_indices=idx)

        cfn = tk.compile(fn, shapeless=True)

        self.assertEqual(cfn(x4).shape, fn(x4).shape)
        self.assertEqual(cfn(x8).shape, fn(x8).shape)

    def test_compile_with_constant(self):
        # Test float
        @partial(tk.compile)
        def fun(x, y):
            return x + y

        z = fun(tk.array(1.0), 1.0)
        self.assertEqual(z.item(), 2.0)

        z = fun(tk.array(1.0), 2.0)
        self.assertEqual(z.item(), 3.0)

        z = fun(tk.array(1.0), y=1.0)
        self.assertEqual(z.item(), 2.0)

        z = fun(tk.array(1.0), y=3.0)
        self.assertEqual(z.item(), 4.0)

        # Test tuple
        @partial(tk.compile)
        def fun(x, y=(1, 2)):
            return x + y[0] + y[1]

        z = fun(tk.array(1))
        self.assertEqual(z.item(), 4)

        z = fun(tk.array(1), (2, 2))
        self.assertEqual(z.item(), 5)

        z = fun(tk.array(1), (2, 1))
        self.assertEqual(z.item(), 4)

        # Test bool
        @partial(tk.compile)
        def fun(x, y):
            if y:
                return x + 1
            else:
                return x + 2

        z = fun(tk.array(1), True)
        self.assertEqual(z.item(), 2)

        z = fun(tk.array(1), False)
        self.assertEqual(z.item(), 3)

        # Test string
        @partial(tk.compile)
        def fun(x, y):
            if y == "one":
                return x + 1
            else:
                return x + 2

        z = fun(tk.array(1), "one")
        self.assertEqual(z.item(), 2)

        z = fun(tk.array(1), "two")
        self.assertEqual(z.item(), 3)

        # Test nested constant
        @partial(tk.compile)
        def fun(x, y):
            if y[0][0] == 1:
                return x + 1
            else:
                return x + 2

        z = fun(tk.array(1), [[1]])
        self.assertEqual(z.item(), 2)

        z = fun(tk.array(1), [[0]])
        self.assertEqual(z.item(), 3)

        @partial(tk.compile)
        def fun(x, a, b):
            for ai in a:
                for bi in b:
                    x = bi * x + ai
            return x

        z = fun(tk.array(1), [1, 1], [2])
        self.assertEqual(z.item(), 7)

        z = fun(tk.array(1), [1], [1, 2])
        self.assertEqual(z.item(), 5)

        counter = [0]

        @partial(tk.compile)
        def fun(x, y):
            counter[0] += 1
            return x + y

        z = fun(tk.array(1), 1)
        self.assertEqual(z.item(), 2)

        z = fun(1, tk.array(1))
        self.assertEqual(z.item(), 2)

        self.assertEqual(counter[0], 2)

        y = 1.0

        @tk.compile
        def fun(x, constant):
            return x + y

        constant1 = "abc"
        out = fun(tk.array(0.0), constant1)
        self.assertEqual(out, tk.array(1.0))

        # new object, same value, no recompilation
        y = 2.0
        constant2 = "abc".encode("utf-8").decode("utf-8")
        out = fun(tk.array(0.0), constant2)
        self.assertEqual(out, tk.array(1.0))

        # same object, new value, recompilation
        constant2 = "xyz"
        out = fun(tk.array(0.0), constant2)
        self.assertEqual(out, tk.array(2.0))

    def test_compile_inf(self):
        @tk.compile
        def fun(x):
            return tk.isinf(x + 2)

        out = fun(tk.array([0.0]))
        self.assertEqual(out.item(), False)

    def test_unsupported_input_types(self):
        class MyClass:
            value = 1

        @tk.compile
        def fun(x, y):
            return x + y.value

        with self.assertRaises(ValueError):
            out = fun(tk.array(0.0), MyClass())

        with self.assertRaises(ValueError):
            out = fun(tk.array(0.0), y=MyClass())

    def test_compile_create_list(self):
        @tk.compile
        def fun():
            return [0.1 * tk.zeros((2,)), 0.1 * tk.zeros((2,))]

        out = fun()
        tk.eval(out)

    def test_compile_vjp(self):
        def fun(w):
            w1 = w + w
            w2 = w + w
            return w @ w1 + w2 @ w2

        def step(w):
            out, grad = tk.vjp(fun, (w,), (tk.array([[1.0, 1.0], [1.0, 1.0]]),))
            return out[0], grad[0]

        w = tk.zeros((2, 2))
        tk.eval(w)

        expected = step(w)
        out = tk.compile(step)(w)
        self.assertTrue(tk.allclose(expected[0], out[0]))
        self.assertTrue(tk.allclose(expected[1], out[1]))

        def fun(w1, w2, x):
            x = x @ w1
            y = x @ w2
            x = x + y * y
            return (x * x).sum()

        w1 = tk.zeros((4, 4))
        w2 = tk.zeros((4, 4))
        x = tk.zeros((4, 4))

        def step(w1, w2, x):
            loss, gradient = tk.value_and_grad(fun)(w1, w2, x)
            w1 = w1 + gradient
            return loss, w1

        tk.eval(x, w1, w2)
        expected = step(w1, w2, x)
        out = tk.compile(step)(w1, w2, x)

        self.assertTrue(tk.allclose(expected[0], out[0]))
        self.assertTrue(tk.allclose(expected[1], out[1]))

    def test_shapeless_mean(self):
        def mean(x):
            return tk.mean(x, keepdims=True)

        cfun = tk.compile(mean)
        out = cfun(tk.ones((5, 5)))
        self.assertTrue(tk.allclose(out, tk.array(1.0)))

        cmean = tk.compile(mean, shapeless=True)

        x = tk.ones(2)
        out = cmean(x)
        self.assertTrue(tk.allclose(out, mean(x)))

        x = tk.ones(4)
        out = cmean(x)
        self.assertTrue(tk.allclose(out, mean(x)))

        x = tk.ones(7)
        out = cmean(x)
        self.assertTrue(tk.allclose(out, mean(x)))

    def test_compile_broadcast_only(self):
        def fn(a):
            a = tk.broadcast_to(a, (1,))
            return a + a

        out = tk.compile(fn)(tk.array(2.0))
        # Make sure repr can be called
        self.assertTrue(repr(out) is not None)
        self.assertTrue(tk.array_equal(out, tk.array([4.0])))

    def test_compile_with_long_name(self):
        def fn(a, b):
            for _ in range(10):
                a = a - 1.0
                b = b - 1.0
            return a + b

        out = tk.compile(fn)(tk.array(10.0), tk.array(20.0))
        self.assertEqual(out.item(), 10.0)

    def test_compile_multi_output(self):
        def fn(x):
            ys = [x]
            for i in range(5):
                ys.append(ys[-1] + x)
            return ys, tk.sum(ys[-1])

        x = tk.ones(1, dtype=tk.int32)
        y1 = tk.compile(fn)(x)[1]
        y2 = fn(x)[1]
        self.assertEqual(y1.item(), y2.item())
        self.assertEqual(y1.item(), 6)

    def test_inf_constant(self):
        def fn(x):
            return tk.where(tk.isinf(x), 0, 1)

        x = tk.array([0, float("inf"), 1], dtype=tk.bfloat16)
        self.assertTrue(tk.array_equal(tk.compile(fn)(x), fn(x)))

    def test_max_into_equal(self):
        x = tk.random.uniform(shape=(1, 2, 2))
        tk.eval(x)

        def fn():
            maxes = tk.max(x, axis=(1, 2), keepdims=True)
            return x == maxes

        out = tk.compile(fn)()
        expected = fn()
        self.assertTrue(tk.array_equal(expected, out))

    def test_dtypes(self):
        x = tk.array([0, 1, 2, 3])
        dtypes = [tk.bool_, tk.int8, tk.uint8, tk.int16, tk.uint16]
        for dtype in dtypes:
            x = x.astype(dtype)
            tk.eval(x)

            def fn(x):
                return x * 1 + 0

            out = tk.compile(fn)(x)
            expected = fn(x)
            self.assertTrue(tk.array_equal(expected, out))

    def test_compile_without_captured_inputs(self):
        x = tk.array([1, 2, 3]) + 2

        def fn(a):
            y = x + 1
            return a + y

        with self.assertRaises(ValueError):
            y = tk.compile(fn)(x)

        x = tk.array([1.0, 2.0]) + tk.array([1.0, 2.0])
        y = None

        def fn(x):
            nonlocal y
            if y is None:
                y = tk.array([1.0, 2.0])

            y = y + x
            return y

        fn(x)
        with self.assertRaises(ValueError):
            y = tk.compile(fn)(x)

    def test_compile_dynamic_dims(self):
        a = tk.random.uniform(shape=(2,) * 10)
        b = tk.random.uniform(shape=(2,) * 10)
        a = a.T
        tk.eval(a, b)

        def fn(a, b):
            return tk.abs(a + b)

        out = tk.compile(fn)(a, b)
        expected = fn(a, b)
        self.assertTrue(tk.allclose(out, expected))

    def test_compile_many_inputs(self):
        inputs = [tk.ones((2, 2, 2, 2)) for _ in range(20)]
        inputs[0] = inputs[0].T

        @tk.compile
        def fun(*inputs):
            x = inputs[0]
            for y in inputs[1:10]:
                x = x + y
            a = inputs[10]
            for b in inputs[11:]:
                a = a + b
            return x + a

        out = fun(*inputs)
        self.assertTrue(tk.allclose(out, tk.full((2, 2), 20)))

        @tk.compile
        def fun(arrs):
            for _ in range(6):
                arrs = [x + y for x, y in zip(arrs[::2], arrs[1::2])]
            return arrs[0]

        arrs = [tk.array([1.0, 2.0]) for _ in range(64)]
        out = fun(arrs)
        self.assertTrue(tk.allclose(out, tk.array([64.0, 128.0])))

        inputs = [tk.arange(16384).astype(tk.float16) for _ in range(8)]

        def fun(inputs):
            a = inputs[0] + inputs[1]
            b = inputs[2] + inputs[3]
            c = inputs[4] + inputs[5]
            d = inputs[6] + inputs[7]
            return a * b * c * d

        out = tk.compile(fun)(inputs)
        expected = fun(inputs)
        self.assertTrue(tk.allclose(out, expected))

    def test_compile_many_outputs(self):
        @tk.compile
        def fun(arr):
            arrs = [arr] * 64
            first_arrs = None
            for _ in range(6):
                arrs = [x + y for x, y in zip(arrs[::2], arrs[1::2])]
                if first_arrs is None:
                    first_arrs = arrs
            return arrs[0], first_arrs

        out = fun(tk.array([1.0, 2.0]))
        self.assertTrue(tk.allclose(out[0], tk.array([64.0, 128.0])))

    def test_shapeless_compile_matmul(self):
        a = tk.array([0.0, 1.0, 2.0])
        b = tk.array([0.0, 1.0, 2.0])

        fun = tk.compile(lambda a, b: a @ b, shapeless=True)
        self.assertTrue(tk.allclose(fun(a, b), a @ b))

    def test_shapeless_compile_addmm(self):
        def fun(c, a, b):
            return tk.addmm(c, a, b)

        cfun = tk.compile(fun, shapeless=True)

        # First shape
        c = tk.ones((2, 4))
        a = tk.ones((2, 3))
        b = tk.ones((3, 4))
        self.assertTrue(tk.allclose(cfun(c, a, b), fun(c, a, b)))

        # Different shape, same ranks — should not recompile
        c = tk.ones((3, 5))
        a = tk.ones((3, 6))
        b = tk.ones((6, 5))
        self.assertTrue(tk.allclose(cfun(c, a, b), fun(c, a, b)))

        # With alpha and beta
        fun2 = tk.compile(
            lambda c, a, b: tk.addmm(c, a, b, alpha=2.0, beta=3.0), shapeless=True
        )
        c = tk.ones((2, 4))
        a = tk.ones((2, 3))
        b = tk.ones((3, 4))
        expected = 3.0 * c + 2.0 * (a @ b)
        self.assertTrue(tk.allclose(fun2(c, a, b), expected))

    def test_shapeless_compile_slice_update(self):
        def fun(x):
            x[2] = tk.array([3.0])
            return x

        cfun = tk.compile(fun, shapeless=True)

        a = tk.array([0.0, 1.0, 2.0, 3.0])
        self.assertTrue(tk.allclose(cfun(a), fun(a)))

        a = tk.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertTrue(tk.allclose(cfun(a), fun(a)))

    def test_shapeless_compile_with_reshape(self):
        def fun(x):
            return x.reshape(x.shape[0] * x.shape[1], -1)

        compiled_fun = tk.compile(fun, shapeless=True)

        x = tk.zeros(shape=(2, 3, 4))
        out = compiled_fun(x)
        self.assertEqual(out.shape, (6, 4))

        x = tk.zeros(shape=(2, 3, 8))
        out = compiled_fun(x)
        self.assertEqual(out.shape, (6, 8))

        x = tk.zeros(shape=(5, 5, 5))

        with self.assertRaises(ValueError):
            compiled_fun(x)

    def test_compile_shapeless_with_broadcast(self):
        a = tk.array(0.0)
        b = tk.ones((2, 2))

        def fun(a):
            return tk.broadcast_to(a, b.shape)

        cfun = tk.compile(fun, shapeless=True)
        # Works on the first shape
        cfun(a)

        # Fails on a different shape
        with self.assertRaises(ValueError):
            cfun(tk.array(0.0).reshape(1, 1, 1))

        def fun(a, b):
            return tk.broadcast_arrays(a, b)

        cfun = tk.compile(fun, shapeless=True)
        a, b = cfun(a, b)
        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(b.shape, (2, 2))

        # Batched matmul
        a = tk.zeros((2, 1, 4, 2))
        b = tk.zeros((3, 2, 5))

        def fun(a, b):
            return a @ b

        cfun = tk.compile(fun, shapeless=True)
        out = cfun(a, b)
        self.assertEqual(out.shape, (2, 3, 4, 5))

        # Shapeless compile should be preserved over vjp, jvp, vmap
        def fun(args):
            return sum(args).sum()

        a = tk.array(0.0)
        b = tk.ones((2, 2))

        cfun = tk.compile(tk.grad(fun), shapeless=True)
        out = cfun((a, b))

        self.assertEqual(out[0].shape, ())
        self.assertEqual(out[1].shape, (2, 2))

        out = cfun((b, a))

        self.assertEqual(out[0].shape, (2, 2))
        self.assertEqual(out[1].shape, ())

        # Shapeless compile should be preserved over vjp, jvp, vmap
        def fun(args):
            return (args[0] @ args[1]).sum()

        a = tk.zeros((2, 1, 4, 2))
        b = tk.zeros((3, 2, 5))

        cfun = tk.compile(tk.grad(fun), shapeless=True)
        out = cfun((a, b))

        self.assertEqual(out[0].shape, (2, 1, 4, 2))
        self.assertEqual(out[1].shape, (3, 2, 5))

        a = tk.zeros((3, 1, 4, 2))
        b = tk.zeros((2, 2, 5))

        out = cfun((a, b))

        self.assertEqual(out[0].shape, (3, 1, 4, 2))
        self.assertEqual(out[1].shape, (2, 2, 5))

    def test_shapeless_compile_reduce_after_gather(self):
        # Reductions over dimensions that happen to have size 1 at trace time
        # used to be elided from the graph, so replays with larger dynamic
        # shapes returned stale values (issue #3201)
        buf = tk.array([10.0, 20.0, 30.0, 40.0, 50.0])
        reductions = [
            tk.sum,
            tk.mean,
            tk.prod,
            tk.min,
            tk.max,
            tk.all,
            tk.any,
            tk.argmin,
            tk.argmax,
        ]
        for reduction in reductions:

            def fun(buf, idx):
                return reduction(tk.take(buf, idx, axis=0))

            # Trace with a size-1 reduction and replay with larger sizes
            cfun = tk.compile(fun, shapeless=True)
            for n in [1, 2, 3, 4]:
                idx = tk.arange(n)
                self.assertTrue(
                    tk.array_equal(cfun(buf, idx), fun(buf, idx)),
                    f"{reduction.__name__} failed for n={n}",
                )

            # Replay with size 1 so the reduction is an identity at runtime
            cfun = tk.compile(fun, shapeless=True)
            for n in [2, 1]:
                idx = tk.arange(n)
                self.assertTrue(
                    tk.array_equal(cfun(buf, idx), fun(buf, idx)),
                    f"{reduction.__name__} failed for replay with n={n}",
                )

    def test_leaks(self):
        gc.collect()
        if tk.metal.is_available():
            mem_pre = tk.get_active_memory()
        else:
            mem_pre = 0

        def outer():
            d = {}

            def f(x):
                return d["x"]

            d["f"] = tk.compile(f)
            d["x"] = tk.array([0] * 1000)

        for _ in range(5):
            outer()
            gc.collect()

        if tk.metal.is_available():
            mem_post = tk.get_active_memory()
        else:
            mem_post = 0

        self.assertEqual(mem_pre, mem_post)

    def test_double_constant(self):
        with tk.stream(tk.cpu):
            x = tk.array(1.0, dtype=tk.float64)

            def fun(x):
                return (x + math.pi) * 2.0

            y = fun(x).item()
            y_compiled = tk.compile(fun)(x).item()
            self.assertEqual(y, y_compiled)

    def test_shared_broadcast(self):
        def fun(x, y, z):
            yy = tk.broadcast_to(y, z.shape)
            return (x + yy * z), yy.sum()

        a = tk.random.normal((10, 10))
        b = tk.array(0.1)
        c = tk.random.normal((10, 10))
        tk.eval(a, b, c)
        fc = tk.compile(fun)
        d = fc(a, b, c)

        s = StringIO()
        tk.export_to_dot(s, a=a, b=b, c=c, d1=d[0], d2=d[1])
        s.seek(0)
        s = s.read()

        self.assertTrue("CompiledBroadcastMultiplyAdd" in s)
        d_hat = fun(a, b, c)
        self.assertTrue(tk.allclose(d[0], d_hat[0]))
        self.assertTrue(tk.allclose(d[1], d_hat[1]))

    def test_compile_large_graph_with_broadcasts(self):
        N = 20
        _as = [tk.array(2 * i, dtype=tk.float32) for i in range(N)]
        _bs = [tk.array(i, dtype=tk.float32) for i in range(N)]
        _c = tk.array(0.0)
        x = tk.random.normal((2, 2))

        def f(x):
            y = 0
            for i in range(N):
                y = y + _as[i] * x * _bs[i] * _c
            return y

        ref = f(x)
        tk.eval(ref)
        f = tk.compile(f)
        for i in range(2):
            y = f(x)
            tk.eval(y)

        self.assertTrue(tk.allclose(y, ref))

    def test_wrap_compiled(self):
        @tk.compile
        def inner():
            pass

        @wraps(inner)
        def wrapper():
            pass

    def test_compiled_preserves_attributes(self):
        def inner(x: tk.array, y: str):
            """
            A useful function.
            """
            pass

        c_inner = tk.compile(inner)
        self.assertEqual(inner.__name__, c_inner.__name__)
        self.assertEqual(inner.__qualname__, c_inner.__qualname__)
        self.assertEqual(inner.__doc__, c_inner.__doc__)
        self.assertEqual(inspect.signature(inner), inspect.signature(c_inner))

    def test_compile_with_none(self):
        @tk.compile
        def fun(x, y):
            if y is None:
                return tk.abs(x - 2.0)
            else:
                return tk.abs(x + y)

        out = fun(tk.array(1.0), None)
        self.assertEqual(out.item(), 1.0)

        out = fun(tk.array(1.0), tk.array(2.0))
        self.assertEqual(out.item(), 3.0)

    def test_compile_changing_outputs(self):
        @tk.compile
        def fun(x, y):
            if y is None:
                return 2 * x
            elif (
                isinstance(x, tk.array)
                and isinstance(y, tk.array)
                and x.dtype == y.dtype == tk.float32
            ):
                return [x + y]
            elif y.dtype == tk.bool_:
                return {"a": x, "b": y * x}
            else:
                return None

        a = fun(tk.array(1.0), tk.array(2.0))
        self.assertTrue(isinstance(a, list))
        self.assertEqual(a[0].item(), 3.0)

        b = fun(tk.array(1.0), tk.array(True))
        self.assertTrue(isinstance(b, dict))
        self.assertEqual(b["a"].item(), 1.0)
        self.assertEqual(b["b"].item(), 1.0)

        c = fun(tk.array(1.0), None)
        self.assertTrue(isinstance(c, tk.array))
        self.assertEqual(c.item(), 2.0)

        d = fun(False, tk.array(1.0))
        self.assertTrue(d is None)

    def test_compile_changing_outputs_with_state(self):
        state = [tk.array(1.0)]

        @partial(tk.compile, inputs=state, outputs=state)
        def fun(y):
            x = state[0]
            if y.dtype == tk.float32:
                state[0] = 2 * y
                return [x, y, x + y]
            elif y.dtype == tk.int32:
                state[0] *= 2
                return x + y

        for i in range(10):
            fun(tk.array(1.0))
            fun(tk.array(1))

        self.assertEqual(state[0].item(), 4)

    def test_outputs_changing(self):
        @tk.compile
        def fun(x):
            x = tk.abs(tk.negative(x))
            y = tk.abs(x)
            return x, y

        @tk.compile
        def fun2(x):
            x = tk.abs(tk.negative(x))
            y = tk.abs(x)
            return y

        a, b = fun(tk.array(-1.0))
        tk.eval(a, b)

        a = fun2(tk.array(-1.0))
        self.assertEqual(a.item(), 1.0)

    def test_multiple_compile_same_capture(self):
        def fun(do_compile):
            t = tk.ones((10,))
            u = (1.0 - t) * 0.0 + t * 3.0

            o = tk.ones((6,))
            b = o[:, None] * u

            c = b * tk.ones_like(u)

            a = tk.ones((6,))
            if do_compile:
                d = tk.compile(lambda x: x @ b)(a)
                e = tk.compile(lambda x: x @ c.T)(d)
            else:
                d = a @ b
                e = d @ c.T
            return e

        out = fun(True)
        tk.eval(out)
        expected = fun(False)
        self.assertTrue(tk.allclose(out, expected))

    def test_compile_types(self):
        from typing import NamedTuple

        class Vector(tuple):
            pass

        class State(NamedTuple):
            a: tk.array
            b: tk.array

        def transform(x: State):
            return State(x.a + 10, x.b * 10)

        def transform_tuple(t):
            return (t[0] + 10, t[1] * 10)

        def transform_vector(t):
            return Vector([t[0] + 10, t[1] * 10])

        x = State(tk.array(1), tk.array(2))

        compiled_transform = tk.compile(transform)
        compiled_transform_tuple = tk.compile(transform_tuple)
        compiled_transform_vector = tk.compile(transform_vector)

        x_batch_tuple = (tk.array([1, 2, 3]), tk.array([4, 5, 6]))
        out1 = compiled_transform_tuple(x_batch_tuple)

        self.assertTrue(isinstance(out1, tuple))
        self.assertTrue(tk.array_equal(out1[0], tk.array([11, 12, 13])))
        self.assertTrue(tk.array_equal(out1[1], tk.array([40, 50, 60])))

        x_batch = State(tk.array([1, 2, 3]), tk.array([4, 5, 6]))
        out2 = compiled_transform(x_batch)
        self.assertTrue(isinstance(out2, State))
        self.assertTrue(tk.array_equal(out2.a, tk.array([11, 12, 13])))
        self.assertTrue(tk.array_equal(out2.b, tk.array([40, 50, 60])))

        x_batch_vector = Vector([tk.array([1, 2, 3]), tk.array([4, 5, 6])])
        out3 = compiled_transform_vector(x_batch_vector)
        self.assertTrue(isinstance(out3, Vector))
        self.assertTrue(tk.array_equal(out3[0], tk.array([11, 12, 13])))
        self.assertTrue(tk.array_equal(out3[1], tk.array([40, 50, 60])))

    def test_compile_output_with_siblings(self):
        @tk.compile
        def fun(x, y):
            return tk.divmod(tk.abs(x), tk.abs(y))[0]

        out = fun(tk.array(1.0), tk.array(1.0))
        self.assertEqual(out.item(), 1.0)

        # Make sure the following compiles without issue
        def loss_fn(params, x):
            emb, w = params
            return tk.fast.layer_norm(emb[x], w, None, 1e-4).sum()

        emb = tk.zeros((10, 32))
        w = tk.zeros((32,))

        loss_and_grad_fn = tk.value_and_grad(loss_fn)

        x = tk.zeros(shape=(4, 32), dtype=tk.int32)
        tk.eval(x, emb, w)

        @tk.compile
        def step(emb, w, x):
            loss, grads = loss_and_grad_fn((emb, w), x)
            return loss, grads

        loss, grads = step(emb, w, x)
        tk.eval(loss, grads)

    def test_compile_donates_input_buffer(self):
        tk.set_default_device(tk.cpu)

        def fun(x):
            return tk.sin(x) + 1

        compiled_fn = tk.compile(fun)

        input = tk.arange(16, dtype=tk.float32)
        tk.eval(input)
        in_ptr = np.asarray(input, copy=False).__array_interface__["data"][0]

        out = compiled_fn(input)
        del input  # Ensure the reference is dropped
        tk.eval(out)

        self.assertEqual(
            np.asarray(out, copy=False).__array_interface__["data"][0], in_ptr
        )

    def test_compile_negative_strides(self):
        # 1D negative stride with elementwise expression
        @tk.compile
        def f(x):
            return 2.0 * x[::-1]

        x = tk.arange(8, dtype=tk.float32)
        expected = 2.0 * x[::-1]
        self.assertTrue(tk.array_equal(f(x), expected))

        # 1D negative stride with slice update
        def g_eager(x):
            base = tk.zeros_like(x)
            base[::-1] += 2.0 * x[::-1]
            return base

        g_compiled = tk.compile(g_eager)
        expected = g_eager(x)
        self.assertTrue(tk.array_equal(g_compiled(x), expected))

        # 2D negative stride
        @tk.compile
        def h(x):
            return x[::-1] + 1.0

        y = tk.arange(12, dtype=tk.float32).reshape(3, 4)
        expected = y[::-1] + 1.0
        self.assertTrue(tk.array_equal(h(y), expected))

        # Mixed positive and negative strides
        @tk.compile
        def m(x):
            return x[::-1, ::2] * 3.0

        z = tk.arange(24, dtype=tk.float32).reshape(4, 6)
        expected = z[::-1, ::2] * 3.0
        self.assertTrue(tk.array_equal(m(z), expected))

        # 4D negative stride (exercises work_per_thread > 1 path)
        @tk.compile
        def p(x):
            return x + 1.0

        w = tk.arange(120, dtype=tk.float32).reshape(2, 3, 4, 5)
        expected = w[::-1, :, ::-1, :] + 1.0
        self.assertTrue(tk.array_equal(p(w[::-1, :, ::-1, :]), expected))

    def test_compile_abs_unsigned(self):
        # abs has to compile for the wider unsigned types too
        fun = lambda x: tk.abs(x) + 1
        for dtype in [tk.uint8, tk.uint16, tk.uint32, tk.uint64]:
            x = tk.array([1, 2, 3], dtype)
            self.assertTrue(tk.array_equal(tk.compile(fun)(x), fun(x)))

    def test_compiled_subnormal_bool_cast(self):
        f32_sub = tk.array(np.array([0x00000001] * 4, dtype=np.uint32)).view(tk.float32)
        f16_sub = tk.array(np.array([0x0001] * 4, dtype=np.uint16)).view(tk.float16)
        bf16_sub = tk.array(np.array([0x0001] * 4, dtype=np.uint16)).view(tk.bfloat16)

        # A single-op compile does not fuse; the fused path needs >= 2 ops.
        fn = tk.compile(lambda x: tk.broadcast_to(x, (2, 4)).astype(tk.bool_))
        for sub in (f32_sub, f16_sub, bf16_sub):
            self.assertTrue(tk.all(fn(sub)).item())

    def test_compile_different_log_bases(self):
        # The logs are intermediates, since outputs are not simplified.
        def entropies(p):
            nats = -tk.sum(p * tk.log(p))
            bits = -tk.sum(p * tk.log2(p))
            return tk.stack([nats, bits])

        p = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        expected = np.array(
            [-(p * np.log(p)).sum(), -(p * np.log2(p)).sum()], dtype=np.float32
        )
        out = tk.compile(entropies)(tk.array(p))
        self.assertTrue(np.allclose(out, expected, atol=1e-5))

    def test_compile_equal_nan(self):
        def fun(x):
            return tk.stack(
                [tk.array_equal(x, x), tk.array_equal(x, x, equal_nan=True)]
            )

        x = tk.array([1.0, float("nan"), 3.0])
        self.assertTrue(tk.array_equal(tk.compile(fun)(x), tk.array([False, True])))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
