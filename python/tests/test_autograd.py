# Copyright © 2023 Apple Inc.

import gc
import itertools
import math
import unittest

import numpy as np
import tiki as tk
import tiki_tests

try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False


class TestAutograd(tiki_tests.TIKITestCase):
    def test_jvp(self):
        fun = lambda x: 2 * x
        out, dout = tk.jvp(fun, [tk.array(1.0)], [tk.array(2.0)])
        self.assertEqual(out[0].item(), 2.0)
        self.assertEqual(dout[0].item(), 4.0)

        fun = lambda x, y: x * y
        _, out = tk.jvp(
            fun, [tk.array(4.0), tk.array(2.0)], [tk.array(3.0), tk.array(2.0)]
        )
        self.assertEqual(out[0].item(), 4.0 * 2.0 + 2.0 * 3.0)

        fun = lambda x, y, z: (x * y, y * z)
        _, out = tk.jvp(
            fun,
            [tk.array(2.0), tk.array(4.0), tk.array(6.0)],
            [tk.array(1.0), tk.array(3.0), tk.array(1.0)],
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].item(), 4.0 * 1.0 + 2.0 * 3.0)
        self.assertEqual(out[1].item(), 4.0 * 1.0 + 6.0 * 3.0)

    def test_jvp_comparison_tangent_dtype(self):
        # Comparison op JVP tangents should preserve the input tangent's
        # dtype (e.g. float32), not return bool. Using bool tangents causes
        # downstream ops like negative to crash. (issue #3081)
        x = tk.array([1.0, -2.0, 3.0])
        t = tk.ones_like(x)

        for op in [
            tk.greater,
            tk.less,
            tk.equal,
            tk.greater_equal,
            tk.less_equal,
            tk.not_equal,
        ]:
            _, tangents = tk.jvp(lambda x, _op=op: _op(x, 0.0), [x], [t])
            self.assertEqual(tangents[0].dtype, tk.float32)

    def test_jvp_with_constant_inputs(self):
        # JVPs of primitives with only a subset of inputs traced used to
        # index tangents out of bounds and silently return wrong tangents
        # (issue #3627)

        # d/dt (t + 1)^2 = 2 at t = 0
        cases = [
            lambda t: tk.where((t + 1) ** 2 > -1, (t + 1) ** 2, 999.0),
            lambda t: tk.where((t + 1) ** 2 < -1, 999.0, (t + 1) ** 2),
            lambda t: tk.where(tk.array(True), (t + 1) ** 2, 999.0),
            lambda t: tk.where(tk.array(False), 999.0, (t + 1) ** 2),
        ]
        for fun in cases:
            _, (dout,) = tk.jvp(fun, [tk.array(0.0)], [tk.array(1.0)])
            self.assertEqual(dout.item(), 2.0)

        # Constant condition with both branches traced
        _, (dout,) = tk.jvp(
            lambda a, b: tk.where(tk.array([True, False]), a, b),
            [tk.zeros(2), tk.zeros(2)],
            [tk.array([1.0, 2.0]), tk.array([3.0, 4.0])],
        )
        self.assertTrue(tk.array_equal(dout, tk.array([1.0, 4.0])))

        # The tangent of a where with only the condition traced is zero
        # with the output's dtype
        _, (dout,) = tk.jvp(
            lambda c: tk.where(c > 0, 2.0, 3.0), [tk.array(1.0)], [tk.array(1.0)]
        )
        self.assertEqual(dout.item(), 0.0)
        self.assertEqual(dout.dtype, tk.float32)

        # d/dy atan2(y, x) = x / (x^2 + y^2)
        _, (dout,) = tk.jvp(
            lambda y: tk.arctan2(y, tk.array(2.0)), [tk.array(1.0)], [tk.array(1.0)]
        )
        self.assertAlmostEqual(dout.item(), 0.4, places=6)

        # d/dx atan2(y, x) = -y / (x^2 + y^2)
        _, (dout,) = tk.jvp(
            lambda x: tk.arctan2(tk.array(2.0), x), [tk.array(1.0)], [tk.array(1.0)]
        )
        self.assertAlmostEqual(dout.item(), -0.4, places=6)

        # masked_scatter with a constant destination
        mask = tk.array([True, False, True, False])

        def masked_set(src):
            dst = tk.zeros(4)
            dst[mask] = src
            return dst

        _, (dout,) = tk.jvp(masked_set, [tk.ones(2)], [tk.ones(2)])
        self.assertTrue(tk.array_equal(dout, tk.array([1.0, 0.0, 1.0, 0.0])))

    def test_jvp_through_bitwise_ops(self):
        # JVPs of bitwise ops returned one tangent per traced input instead
        # of one per output which corrupted the tangents of downstream
        # outputs (issue #3629). The corruption is out-of-bounds UB that can
        # go unnoticed in release builds; the assert in the jvp transform
        # catches it deterministically in debug builds.
        def fun(x):
            b = (x == 0) & (x > -1)
            return x + b.astype(tk.float32)

        x = tk.array([0.0, 1.0, 2.0])
        _, (dout,) = tk.jvp(fun, [x], [tk.ones_like(x)])
        self.assertTrue(tk.array_equal(dout, tk.ones_like(x)))

        def fun(x):
            b = (x == 0) | (x > 1)
            return x * b.astype(tk.float32)

        _, (dout,) = tk.jvp(fun, [x], [tk.ones_like(x)])
        self.assertTrue(tk.array_equal(dout, tk.array([1.0, 0.0, 1.0])))

    def test_vjp(self):
        fun = lambda x: 2 * x
        out, dout = tk.vjp(fun, [tk.array(1.0)], [tk.array(2.0)])
        self.assertEqual(out[0].item(), 2.0)
        self.assertEqual(dout[0].item(), 4.0)

        fun = lambda x, y: x * y
        _, dout = tk.vjp(fun, [tk.array(4.0), tk.array(2.0)], [tk.array(3.0)])
        self.assertEqual(dout[0].item(), 6.0)
        self.assertEqual(dout[1].item(), 12.0)

        fun = lambda x, y, z: (x * y, y * z)
        _, out = tk.vjp(
            fun,
            [tk.array(2.0), tk.array(4.0), tk.array(6.0)],
            [tk.array(1.0), tk.array(3.0)],
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].item(), 4.0 * 1.0)
        self.assertEqual(out[1].item(), 2.0 * 1.0 + 6.0 * 3.0)
        self.assertEqual(out[2].item(), 4.0 * 3.0)

    def test_jvp_with_partly_traced_inputs(self):
        # power: each traced input must use its own tangent (issue #3634)
        fun = lambda a, b: a**b
        primals = [tk.array(2.0), tk.array(3.0)]
        dyda = 12.0  # d/da a^b = b * a^(b - 1)
        dydb = math.log(2.0) * 8.0  # d/db a^b = ln(a) * a^b
        _, (j,) = tk.jvp(fun, primals, [tk.array(1.0), tk.array(0.0)])
        self.assertAlmostEqual(j.item(), dyda, places=4)
        _, (j,) = tk.jvp(fun, primals, [tk.array(0.0), tk.array(1.0)])
        self.assertAlmostEqual(j.item(), dydb, places=4)
        _, (j,) = tk.jvp(fun, primals, [tk.array(1.0), tk.array(1.0)])
        self.assertAlmostEqual(j.item(), dyda + dydb, places=4)
        fun = lambda a: a ** tk.array(3.0)
        _, (j,) = tk.jvp(fun, [tk.array(2.0)], [tk.array(1.0)])
        self.assertAlmostEqual(j.item(), dyda, places=4)
        fun = lambda b: tk.array(2.0) ** b
        _, (j,) = tk.jvp(fun, [tk.array(3.0)], [tk.array(1.0)])
        self.assertAlmostEqual(j.item(), dydb, places=4)

        # divmod: one tangent per output, and consuming the second output
        # used to crash (issue #3634)
        def fun(x):
            q, r = tk.divmod(x, tk.array(3.0))
            return q, -r

        _, (dq, dr) = tk.jvp(fun, [tk.array(7.0)], [tk.array(1.0)])
        self.assertEqual(dq.item(), 0.0)
        self.assertEqual(dr.item(), 0.0)

        # slice_update with dynamic start indices and a subset of traced
        # inputs (issue #3634)
        src = tk.zeros(4)
        upd = tk.ones(2)
        start = tk.array([1])
        fun = lambda u: tk.slice_update(src, u, start, axes=[0])
        _, (j,) = tk.jvp(fun, [upd], [tk.ones(2)])
        self.assertEqual(j.tolist(), [0.0, 1.0, 1.0, 0.0])
        fun = lambda s: tk.slice_update(s, upd, start, axes=[0])
        _, (j,) = tk.jvp(fun, [src], [tk.ones(4)])
        self.assertEqual(j.tolist(), [1.0, 0.0, 0.0, 1.0])
        fun = lambda s, u: tk.slice_update(s, u, start, axes=[0])
        _, (j,) = tk.jvp(fun, [src, upd], [tk.ones(4), tk.full(2, 2.0)])
        self.assertEqual(j.tolist(), [1.0, 2.0, 2.0, 1.0])

    def test_grad(self):
        fun = lambda x: x * x

        value, dfdx = tk.value_and_grad(fun)(tk.array(0.5))
        self.assertEqual(value.item(), 0.25)
        self.assertEqual(dfdx.item(), 1.0)

        dfdx = tk.grad(fun)(tk.array(0.5))
        self.assertEqual(dfdx.item(), 1.0)

        df2dx2 = tk.grad(tk.grad(fun))(tk.array(0.5))
        self.assertEqual(df2dx2.item(), 2.0)
        df3dx3 = tk.grad(tk.grad(tk.grad(fun)))(tk.array(0.5))
        self.assertEqual(df3dx3.item(), 0.0)

        fun = lambda x, y: x * y
        x = tk.array(2.0)
        y = tk.array(3.0)
        dfdx = tk.grad(fun, argnums=0)(x, y)
        self.assertEqual(dfdx.item(), 3.0)
        dfdx = tk.grad(fun, argnums=1)(x, y)
        self.assertEqual(dfdx.item(), 2.0)

        # Pass non array args to functions works
        fun = lambda x, y: x
        value, dfdx = tk.value_and_grad(fun)(tk.array(2.0), "hello")
        self.assertEqual(value.item(), 2.0)
        self.assertEqual(dfdx.item(), 1.0)

        dfdx = tk.grad(fun)(tk.array(2.0), "hello")
        self.assertEqual(dfdx.item(), 1.0)

        # Raises when function does not return array
        fun = lambda x: "hello"
        with self.assertRaises(ValueError):
            tk.grad(fun)(tk.array(2.0))

        # Raises for invalid argument number or argument type
        fun = lambda x: x
        with self.assertRaises(ValueError):
            tk.grad(fun, argnums=2)(tk.array(2.0))
        with self.assertRaises(ValueError):
            tk.grad(fun, argnums=-2)(tk.array(2.0))
        with self.assertRaises(ValueError):
            tk.grad(fun)("hello")

        # Raises when output is not a scalar array
        fun = lambda x: tk.sum(x, keepdims=True)
        with self.assertRaises(ValueError):
            tk.grad(fun)(tk.ones((2, 2)))

    def test_grad_trees(self):
        fun = lambda x, y: x * y
        value, dfdx = tk.value_and_grad(fun, (0, 1))(tk.array(0.5), tk.array(2.0))
        self.assertEqual(value.item(), 1.0)
        self.assertTrue(isinstance(dfdx, tuple))
        self.assertEqual(dfdx[0].item(), 2.0)
        self.assertEqual(dfdx[1].item(), 0.5)

        fun = lambda x, y: x * y
        value, dfdx = tk.value_and_grad(fun, 1)(tk.array(0.5), tk.array(2.0))
        self.assertEqual(value.item(), 1.0)
        self.assertEqual(dfdx.item(), 0.5)

        fun = lambda p: p["x"] * p["y"]
        value, dfdx = tk.value_and_grad(fun)({"x": tk.array(0.5), "y": tk.array(2.0)})
        self.assertEqual(value.item(), 1.0)
        self.assertEqual(dfdx["x"].item(), 2.0)
        self.assertEqual(dfdx["y"].item(), 0.5)

        fun = lambda p: p["x"] * p["y"]
        with self.assertRaises(ValueError):
            tk.value_and_grad(fun)({"x": 0.5, "y": tk.array(2.0)})
        with self.assertRaises(ValueError):
            tk.value_and_grad(fun, (0, 1))({"x": tk.array(0.5), "y": tk.array(2.0)})

        fun = lambda p, b: tk.square(p[0]["foo"][2]) * b
        value, dfdx = tk.value_and_grad(fun)(
            [{"foo": [[], [], tk.array(2.0)]}], tk.array(0.5)
        )
        self.assertEqual(value.item(), 2.0)
        self.assertEqual(dfdx[0]["foo"][2].item(), 2.0)

        fun = lambda x: x
        with self.assertRaises(TypeError):
            tk.value_and_grad(fun, (None, None))
        with self.assertRaises(ValueError):
            tk.value_and_grad(fun, tuple())
        with self.assertRaises(ValueError):
            tk.grad(fun, argnums=(0, 0))

    def test_auxiliary_values(self):
        def fun(x, y):
            l = (x * y).sum()
            extra = {"loss": l, "foo": y.square() + x.square(), "bar": [1, 2, 3, y, x]}
            return l, extra

        fun_value_grad = tk.value_and_grad(fun)
        fun_grad = tk.grad(fun)

        (loss, a), b = fun_value_grad(tk.ones((2, 2)), tk.ones((2, 2)))
        self.assertEqual(a["loss"].item(), 4)
        self.assertTrue(tk.array_equal(b, tk.ones((2, 2))))
        self.assertTrue(tk.array_equal(a["foo"], 2 * tk.ones((2, 2))))
        self.assertEqual(a["bar"][:3], [1, 2, 3])
        self.assertTrue(tk.array_equal(a["bar"][3], tk.ones((2, 2))))
        self.assertTrue(tk.array_equal(a["bar"][4], tk.ones((2, 2))))

        with self.assertRaises(ValueError):
            _ = fun_grad(tk.ones((2, 2)), tk.ones((2, 2)))

    def test_grad_kwargs(self):
        fun = lambda x, y: x * y
        a, b = tk.array(0.5), tk.array(2.0)
        dfdx = tk.grad(fun)
        self.assertEqual(dfdx(a, b).item(), 2.0)
        self.assertEqual(dfdx(a, y=b).item(), 2.0)
        with self.assertRaises(ValueError):
            dfdx(x=a, y=b).item()

        dfdy = tk.grad(fun, argnums=[], argnames=["y"])
        with self.assertRaises(ValueError):
            dfdy(a, b)
        grads = dfdy(a, y=b)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(grads[0] is None)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["y"].item(), 0.5)
        grads = dfdy(x=a, y=b)
        self.assertEqual(grads[1]["y"].item(), 0.5)
        self.assertEqual(len(grads[1]), 1)

        dfdxy = tk.grad(fun, argnums=[0], argnames=["y"])
        with self.assertRaises(ValueError):
            dfdxy(a, b)
        with self.assertRaises(ValueError):
            dfdxy(x=a, y=b)
        grads = dfdxy(a, y=b)
        self.assertTrue(isinstance(grads, tuple))
        self.assertEqual(grads[0].item(), 2.0)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["y"].item(), 0.5)

        fun = lambda x, y, z: x * y * z
        dfdxyz = tk.grad(fun, argnums=[0, 1], argnames=["z"])
        c = tk.array(4.0)
        grads = dfdxyz(a, b, z=c)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(isinstance(grads[0], tuple))
        self.assertEqual(grads[0][0].item(), 8.0)
        self.assertEqual(grads[0][1].item(), 2.0)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["z"].item(), 1.0)

        fun = lambda x, y: x * y
        dfdy = tk.grad(fun, argnames=["y"])
        grads = dfdy(a, y=b)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(grads[0] is None)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["y"].item(), 0.5)

    def test_captured(self):
        a = tk.array(5.0)
        f = lambda x: a + x
        g = lambda x: a + a
        h = lambda x: x + x

        dfdx = tk.grad(f)
        self.assertEqual(dfdx(a).item(), 1.0)

        dgdx = tk.grad(g)
        self.assertEqual(dgdx(a).item(), 0.0)

        dhdx = tk.grad(h)
        self.assertEqual(dhdx(a).item(), 2.0)

        d2fdx2 = tk.grad(dfdx)
        self.assertEqual(d2fdx2(a).item(), 0.0)

        d2gdx2 = tk.grad(dgdx)
        self.assertEqual(d2gdx2(a).item(), 0.0)

        d2hdx2 = tk.grad(dhdx)
        self.assertEqual(d2hdx2(a).item(), 0.0)

    def test_stop_gradient(self):
        shape_in = (4, 4)
        w_in = tk.ones(shape_in)
        x_in = tk.ones(shape_in)
        cotan = tk.ones(shape_in)

        def h(w, x):
            x1 = 2 * x
            y = tk.stop_gradient(x1)
            y1 = 3 * y
            return w @ y1

        vals, vjps = tk.vjp(h, [w_in, x_in], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], 24.0 * tk.ones(shape_in)))
        self.assertTrue(tk.allclose(vjps[1], tk.zeros(shape_in)))

        g = lambda x: h(w_in, x)
        vals, vjps = tk.vjp(g, [x_in], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.zeros(shape_in)))

    def test_update_state(self):
        y = tk.array([1.0])
        state = tk.zeros((2,))

        def fn(y, x):
            nonlocal state
            x = y * x
            state = state + x
            return x.sum()

        x = tk.ones((2,))
        tk.grad(fn)(y, x)
        tk.eval(state)
        self.assertTrue(tk.allclose(state, tk.ones((2,))))

    def test_scatter_vjp(self):
        def fun(x, idx):
            x[idx] = 2.0
            return x.sum()

        dfdx = tk.grad(fun)(tk.array([1.0, 2.0, 3.0, 4.0]), tk.array([1, 3]))
        self.assertTrue(tk.array_equal(dfdx, tk.array([1.0, 0.0, 1.0, 0.0])))
        self.assertEqual(dfdx.dtype, tk.float32)

        y = tk.array([0.0, 1.0, 2.0, 3.0])

        def fun(x, idx):
            y[idx] = x
            return y.sum()

        dfdx = tk.grad(fun)(tk.array([2.0, 3.0]), tk.array([1, 3]))
        self.assertTrue(tk.array_equal(dfdx, tk.array([1.0, 1.0])))
        self.assertEqual(dfdx.dtype, tk.float32)

    def test_index_vjp_requires_stop_gradient(self):
        msg = "stop_gradient"
        x = tk.array([1.0, 2.0, 3.0, 4.0])
        idx = tk.array([1, 3])
        updates = tk.array([5.0, 6.0])
        x_axis = x[:, None]
        idx_axis = idx[:, None]
        updates_axis = updates[:, None]

        def gather_fun(x, idx):
            return tk.take(x, idx)

        with self.assertRaisesRegex(ValueError, msg):
            tk.vjp(gather_fun, [x, idx], [tk.ones((2,))])

        def gather_axis_fun(x, idx):
            return tk.take_along_axis(x, idx, axis=0)

        with self.assertRaisesRegex(ValueError, msg):
            tk.vjp(gather_axis_fun, [x_axis, idx_axis], [tk.ones((2, 1))])

        def scatter_fun(x, idx, updates):
            return x.at[idx].add(updates)

        with self.assertRaisesRegex(ValueError, msg):
            tk.vjp(scatter_fun, [x, idx, updates], [tk.ones((4,))])

        def scatter_axis_fun(x, idx, updates):
            return tk.put_along_axis(x, idx, updates, axis=0)

        with self.assertRaisesRegex(ValueError, msg):
            tk.vjp(
                scatter_axis_fun,
                [x_axis, idx_axis, updates_axis],
                [tk.ones((4, 1))],
            )

    def test_stop_gradient_computed_indices(self):
        def gather_fun(w):
            idx = tk.stop_gradient(tk.argsort(w)[:2])
            return tk.take(w, idx).sum()

        grad = tk.grad(gather_fun)(tk.array([4.0, 3.0, 2.0, 1.0]))
        self.assertTrue(tk.array_equal(grad, tk.array([0.0, 0.0, 1.0, 1.0])))

        def gather_axis_fun(w):
            idx = tk.stop_gradient(tk.argsort(w, axis=1)[:, :1])
            return tk.take_along_axis(w, idx, axis=1).sum()

        grad = tk.grad(gather_axis_fun)(tk.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]))
        self.assertTrue(
            tk.array_equal(
                grad,
                tk.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            )
        )

        def scatter_fun(w):
            idx = tk.stop_gradient(tk.argsort(w)[:2])
            out = tk.zeros((4,))
            out[idx] = w[:2]
            return out.sum()

        grad = tk.grad(scatter_fun)(tk.array([4.0, 3.0, 2.0, 1.0]))
        self.assertTrue(tk.array_equal(grad, tk.array([1.0, 1.0, 0.0, 0.0])))

        def scatter_axis_fun(w):
            idx = tk.stop_gradient(tk.argsort(w, axis=1)[:, :1])
            updates = w.sum(axis=1, keepdims=True)
            out = tk.put_along_axis(tk.zeros((3, 3)), idx, updates, axis=1)
            return out.sum()

        grad = tk.grad(scatter_axis_fun)(tk.ones((3, 3)))
        self.assertTrue(tk.array_equal(grad, tk.ones((3, 3))))

    def test_take_along_axis_complex_vjp(self):
        x = tk.zeros((4,), dtype=tk.complex64)
        indices = tk.array([3, 3], dtype=tk.int32)
        cotangent = tk.array([1 + 2j, 3 + 4j], dtype=tk.complex64)
        _, (gradient,) = tk.vjp(
            lambda z: tk.take_along_axis(z, indices, axis=0),
            [x],
            [cotangent],
        )
        tk.eval(gradient)
        self.assertEqualArray(
            gradient,
            tk.array([0j, 0j, 0j, 4 + 6j], dtype=tk.complex64),
            atol=0,
            rtol=0,
        )

    def test_scatter_add_vjp(self):
        def fun(src, updates):
            x = src.at[tk.array([1, 3])].add(updates)
            return x

        cotan = tk.array([4.0, 5.0, 6.0, 7.0])
        updates = tk.array([1.0, 2.0])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 5.0, 6.0, 7.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([5.0, 7.0])))

    def test_scatter_max_vjp(self):
        def fun(src, updates):
            x = src.at[tk.array([1, 3])].maximum(updates)
            return x

        cotan = tk.array([4.0, 5.0, 6.0, 7.0])
        updates = tk.array([1.0, 2.0])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 5.0, 6.0, 7.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([0.0, 0.0])))

        updates = tk.array([5.0, 6.0])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 0.0, 6.0, 0.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([5.0, 7.0])))

    def test_scatter_min_vjp(self):
        def fun(src, updates):
            x = src.at[tk.array([1, 3])].minimum(updates)
            return x

        cotan = tk.array([4.0, 5.0, 6.0, 7.0])
        updates = tk.array([5.0, 6.0])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 5.0, 6.0, 7.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([0.0, 0.0])))

        updates = tk.array([1.0, 1.0])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 0.0, 6.0, 0.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([5.0, 7.0])))

    def test_slice_update_max_vjp(self):
        def fun(src, updates):
            x = src.at[1:3].maximum(updates)
            return x

        cotan = tk.array([4.0, 5.0, 6.0, 7.0])
        updates = tk.array([[1.0, 2.0]])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 5.0, 6.0, 7.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([[0.0, 0.0]])))

        updates = tk.array([[5.0, 6.0]])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 0.0, 0.0, 7.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([[5.0, 6.0]])))

    def test_slice_update_min_vjp(self):
        def fun(src, updates):
            x = src.at[1:3].minimum(updates)
            return x

        cotan = tk.array([4.0, 5.0, 6.0, 7.0])
        updates = tk.array([[5.0, 6.0]])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 5.0, 6.0, 7.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([[0.0, 0.0]])))

        updates = tk.array([[1.0, 1.0]])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 0.0, 0.0, 7.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([[5.0, 6.0]])))

    def test_slice_update_add_vjp(self):
        def fun(src, updates):
            x = src.at[1:3].add(updates)
            return x

        cotan = tk.array([4.0, 5.0, 6.0, 7.0])
        updates = tk.array([[1.0, 2.0]])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 5.0, 6.0, 7.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([[5.0, 6.0]])))

    def test_slice_update_multiply_vjp(self):
        def fun(src, updates):
            x = src.at[1:3].multiply(updates)
            return x

        cotan = tk.array([4.0, 5.0, 6.0, 7.0])
        updates = tk.array([[2.0, 3.0]])
        _, vjps = tk.vjp(fun, [tk.array([1.0, 2.0, 3.0, 4.0]), updates], [cotan])
        tk.eval(vjps)

        self.assertTrue(tk.allclose(vjps[0], tk.array([4.0, 10.0, 18.0, 7.0])))
        self.assertTrue(tk.allclose(vjps[1], tk.array([[10.0, 18.0]])))

    def test_split_against_slice(self):
        def f_split(x):
            a, _, b = x.split(3, -1)
            return (a * b).sum()

        def f_slice(x):
            step = x.shape[-1] // 3
            a = x[..., :step]
            b = x[..., -step:]
            return (a * b).sum()

        x = tk.random.uniform(shape=(100, 300))
        tk.eval(x)

        df1 = tk.grad(f_split)
        df2 = tk.grad(f_slice)

        self.assertTrue(tk.allclose(df1(x), df2(x)))

    def test_vjp_types(self):
        def fun(x):
            return x

        for t in [tk.float16, tk.bfloat16, tk.float32]:
            out = tk.grad(fun)(tk.array(1.0, t))
            self.assertEqual(out.dtype, t)

        def fun(x):
            return x.sum()

        for t in [tk.float16, tk.bfloat16, tk.float32]:
            out = tk.grad(fun)(tk.array(1.0, t))
            self.assertEqual(out.dtype, t)

        def fun(x, y):
            return (x + y).sum()

        for t in [tk.float16, tk.bfloat16, tk.float32]:
            out = tk.grad(fun)(tk.array(1.0, t), tk.array(1.0, t))
            self.assertEqual(out.dtype, t)

    def test_power_grad(self):
        x = tk.array(0.0)
        g = tk.grad(lambda x: x**2)(x)
        self.assertEqual(g.item(), 0.0)

        x = tk.array(0.0)
        g = tk.grad(lambda x: x**1.5)(x)
        self.assertEqual(g.item(), 0.0)

        x = tk.array(2.0)
        g = tk.grad(lambda x: x**2)(x)
        self.assertAlmostEqual(g.item(), 4.0)

    def test_eval_in_grad(self):
        arr = tk.array([1.0])
        cotan = tk.array([1.0, 1.0])
        y = tk.array([2.0, 2.0])

        def func(x):
            x = x + y
            cond = x < 1
            cond.tolist()
            return x**2

        _, vjps = tk.vjp(func, (arr,), (cotan,))
        self.assertEqual(vjps[0].item(), 12.0)

        def func(x):
            x = x + tk.array([1.0, 1.0])
            tk.eval(x)
            return x**2

        _, vjps = tk.vjp(func, (arr,), (cotan,))
        self.assertEqual(vjps[0].item(), 8.0)

    def test_power_grad(self):
        def fun(x, y):
            res = x - y
            return res**x

        grad = tk.grad(fun)(tk.array(1.0), tk.array(1.0))
        self.assertEqual(grad.item(), 1.0)

    def test_cumprod_grad(self):
        def fun(y):
            return tk.cumprod(y).sum()

        y = tk.array([2.0, 1.0, 2.0, 2.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([20.0, 38.0, 18.0, 16.0, 8.0])
        self.assertTrue(tk.allclose(out, expected))

        y = tk.array([2.0, 0.0, 2.0, 2.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([1.0, 38.0, 0.0, 0.0, 0.0])
        self.assertTrue(tk.allclose(out, expected))

        y = tk.array([2.0, 0.0, 2.0, 0.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([1.0, 6.0, 0.0, 0.0, 0.0])
        self.assertTrue(tk.allclose(out, expected))

        def fun(y):
            return tk.cumprod(y, inclusive=False).sum()

        y = tk.array([2.0, 1.0, 2.0, 2.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([8.0, 14.0, 6.0, 4.0, 0.0])
        self.assertTrue(tk.allclose(out, expected))

        y = tk.array([2.0, 0.0, 2.0, 2.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([1.0, 14.0, 0.0, 0.0, 0.0])
        self.assertTrue(tk.allclose(out, expected))

        y = tk.array([2.0, 0.0, 2.0, 0.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([1.0, 6.0, 0.0, 0.0, 0.0])
        self.assertTrue(tk.allclose(out, expected))

        def fun(y):
            return tk.cumprod(y, inclusive=False, reverse=True).sum()

        y = tk.array([2.0, 1.0, 2.0, 2.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([0.0, 12.0, 12.0, 15.0, 11.0])
        self.assertTrue(tk.allclose(out, expected))

        y = tk.array([2.0, 0.0, 2.0, 2.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([0.0, 12.0, 6.0, 9.0, 7.0])
        self.assertTrue(tk.allclose(out, expected))

        y = tk.array([2.0, 0.0, 2.0, 0.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([0.0, 0.0, 0.0, 9.0, 1.0])
        self.assertTrue(tk.allclose(out, expected))

        def fun(y):
            return tk.cumprod(y, reverse=True).sum()

        y = tk.array([2.0, 1.0, 2.0, 2.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([12.0, 36.0, 24.0, 27.0, 19.0])
        self.assertTrue(tk.allclose(out, expected))

        y = tk.array([2.0, 0.0, 2.0, 2.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([0.0, 36.0, 6.0, 9.0, 7.0])
        self.assertTrue(tk.allclose(out, expected))

        y = tk.array([2.0, 0.0, 2.0, 0.0, 3.0])
        out = tk.grad(fun)(y)
        expected = tk.array([0.0, 0.0, 0.0, 9.0, 1.0])
        self.assertTrue(tk.allclose(out, expected))

    def test_cummax_grad(self):
        # Ties route to the latest occurrence, matching the cummax indices.
        a = tk.array([3.0, 3.0, 1.0, 5.0, 5.0])

        def fun(y):
            return tk.cummax(y).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(a), tk.array([1.0, 2.0, 0.0, 1.0, 1.0]))
        )

        def fun(y):
            return tk.cummax(y, inclusive=False).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(a), tk.array([1.0, 2.0, 0.0, 1.0, 0.0]))
        )

        def fun(y):
            return tk.cummax(y, reverse=True).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(a), tk.array([0.0, 0.0, 0.0, 4.0, 1.0]))
        )

        def fun(y):
            return tk.cummax(y, reverse=True, inclusive=False).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(a), tk.array([0.0, 0.0, 0.0, 3.0, 1.0]))
        )

        # Non-uniform cotangents are routed to the owning index.
        cot = tk.array([10.0, 1.0, 1.0, 100.0, 1000.0])
        _, vjps = tk.vjp(lambda y: tk.cummax(y), (a,), (cot,))
        self.assertTrue(tk.allclose(vjps[0], tk.array([10.0, 2.0, 0.0, 100.0, 1000.0])))

        # 2D along an inner axis.
        m = tk.array([[1.0, 3.0, 3.0], [4.0, 2.0, 4.0]])

        def fun(y):
            return tk.cummax(y, axis=1).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(m), tk.array([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]]))
        )

        def fun(y):
            return tk.cummax(y, axis=1, inclusive=False).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(m), tk.array([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]]))
        )

    def test_cummin_grad(self):
        a = tk.array([3.0, 3.0, 1.0, 5.0, 5.0])

        def fun(y):
            return tk.cummin(y).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(a), tk.array([1.0, 1.0, 3.0, 0.0, 0.0]))
        )

        def fun(y):
            return tk.cummin(y, inclusive=False).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(a), tk.array([1.0, 1.0, 2.0, 0.0, 0.0]))
        )

        def fun(y):
            return tk.cummin(y, reverse=True).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(a), tk.array([0.0, 0.0, 3.0, 1.0, 1.0]))
        )

        def fun(y):
            return tk.cummin(y, reverse=True, inclusive=False).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(a), tk.array([0.0, 0.0, 2.0, 1.0, 1.0]))
        )

        cot = tk.array([10.0, 1.0, 1.0, 100.0, 1000.0])
        _, vjps = tk.vjp(lambda y: tk.cummin(y), (a,), (cot,))
        self.assertTrue(tk.allclose(vjps[0], tk.array([10.0, 1.0, 1101.0, 0.0, 0.0])))

        # 2D along the outer axis.
        m = tk.array([[1.0, 3.0, 3.0], [4.0, 2.0, 4.0]])

        def fun(y):
            return tk.cummin(y, axis=0).sum()

        self.assertTrue(
            tk.allclose(tk.grad(fun)(m), tk.array([[2.0, 1.0, 2.0], [0.0, 1.0, 0.0]]))
        )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_cummax_cummin_grad_vs_torch(self):
        # Cross-check the cumulative max/min VJP against PyTorch autograd over
        # axes, scan direction, inclusive/exclusive modes, ties, and weighted
        # cotangents. Torch has no reverse or exclusive scan, so reverse is
        # emulated by flipping along the axis and exclusive by shifting the
        # inclusive scan one step (its leading element carries no gradient).
        def torch_scan(x, axis, reverse, inclusive, op):
            xf = torch.flip(x, [axis]) if reverse else x
            scan = torch.cummax if op == "max" else torch.cummin
            c = scan(xf, axis).values
            if not inclusive:
                n = c.size(axis)
                head = torch.zeros_like(c.narrow(axis, 0, 1))  # constant, no grad
                c = torch.cat([head, c.narrow(axis, 0, n - 1)], dim=axis)
            return torch.flip(c, [axis]) if reverse else c

        def mx_scan(z, axis, reverse, inclusive, op):
            scan = tk.cummax if op == "max" else tk.cummin
            return scan(z, axis=axis, reverse=reverse, inclusive=inclusive)

        inputs = [
            np.array([3.0, 3.0, 1.0, 5.0, 5.0, 1.0, 5.0, 2.0], dtype=np.float32),
            np.array(
                [[1.0, 3.0, 3.0, 2.0], [4.0, 2.0, 4.0, 4.0], [4.0, 3.0, 1.0, 4.0]],
                dtype=np.float32,
            ),
        ]
        rng = np.random.default_rng(0)
        for x_np in inputs:
            cotangents = {
                "ones": np.ones_like(x_np),
                "weighted": rng.uniform(1.0, 9.0, x_np.shape).astype(np.float32),
            }
            for axis in range(x_np.ndim):
                for op, reverse, inclusive in itertools.product(
                    ("max", "min"), (False, True), (True, False)
                ):
                    for cot_name, cot_np in cotangents.items():
                        with self.subTest(
                            shape=x_np.shape,
                            axis=axis,
                            op=op,
                            reverse=reverse,
                            inclusive=inclusive,
                            cotangent=cot_name,
                        ):
                            _, (mx_grad,) = tk.vjp(
                                lambda z: mx_scan(z, axis, reverse, inclusive, op),
                                (tk.array(x_np),),
                                (tk.array(cot_np),),
                            )

                            xt = torch.tensor(x_np, requires_grad=True)
                            out = torch_scan(xt, axis, reverse, inclusive, op)
                            (out * torch.tensor(cot_np)).sum().backward()

                            self.assertTrue(
                                np.allclose(
                                    np.array(mx_grad), xt.grad.numpy(), atol=1e-5
                                )
                            )

    def test_topk_grad(self):
        a = tk.array([[1, 2, 6, 4, 5], [9, 5, 6, 7, 8]], tk.float32)

        def fun(x):
            return tk.topk(x, 2)

        out = tk.vjp(fun, (a,), (tk.ones((2, 2)),))[1][0]
        expected = tk.array([[0, 0, 1, 0, 1], [1, 0, 0, 0, 1]], tk.float32)
        self.assertTrue(tk.array_equal(out, expected))

    def test_sort_grad(self):
        # Sort permutes the input, so its vjp must scatter the cotangents back
        # to the original positions (the transpose of the permutation), not
        # gather them forward. A non-involutive permutation exposes the bug.
        x = tk.array([3.0, 1.0, 2.0, 5.0, 4.0])
        cotan = tk.array([10.0, 20.0, 30.0, 40.0, 50.0])
        grad = tk.vjp(lambda a: tk.sort(a), (x,), (cotan,))[1][0]
        self.assertTrue(tk.array_equal(grad, tk.array([30.0, 10.0, 20.0, 50.0, 40.0])))

        # vjp must be the transpose of the jvp (adjoint test) along each axis.
        tk.random.seed(0)
        for axis in (0, 1, -1):
            a = tk.random.normal((4, 6))
            v = tk.random.normal(a.shape)
            w = tk.random.normal(a.shape)
            jv = tk.jvp(lambda z: tk.sort(z, axis=axis), (a,), (v,))[1][0]
            jtw = tk.vjp(lambda z: tk.sort(z, axis=axis), (a,), (w,))[1][0]
            self.assertAlmostEqual(
                tk.sum(w * jv).item(), tk.sum(v * jtw).item(), places=4
            )

    def test_logsumexp_grad(self):
        # The jvp of logsumexp reduces along the axis (sum of softmax * tangent),
        # so the tangent it returns must have the reduced output shape, not the
        # input shape.
        x = tk.array([[1.0, 2.0, 3.0], [4.0, 1.0, 0.0]])
        v = tk.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])
        jv = tk.jvp(lambda z: tk.logsumexp(z, axis=-1, keepdims=True), (x,), (v,))[1][0]
        self.assertEqual(jv.shape, (2, 1))
        expected = tk.sum(tk.softmax(x, axis=-1) * v, axis=-1, keepdims=True)
        self.assertTrue(tk.allclose(jv, expected))

        # vjp must be the transpose of the jvp (adjoint test).
        tk.random.seed(0)
        for keepdims in (True, False):
            a = tk.random.normal((4, 6))
            v = tk.random.normal(a.shape)

            def fun(z):
                return tk.logsumexp(z, axis=-1, keepdims=keepdims)

            w = tk.random.normal(fun(a).shape)
            jv = tk.jvp(fun, (a,), (v,))[1][0]
            jtw = tk.vjp(fun, (a,), (w,))[1][0]
            self.assertAlmostEqual(
                tk.sum(w * jv).item(), tk.sum(v * jtw).item(), places=4
            )

    def test_custom_function(self):
        # Make a custom function
        my_exp = tk.custom_function(tk.exp)

        # Ensure everything works
        dy = tk.grad(my_exp)(tk.array(1.0))
        self.assertTrue(tk.allclose(dy, tk.exp(tk.array(1.0))))
        (ex,), (dex,) = tk.jvp(my_exp, [tk.array(1.0)], [tk.array(1.0)])
        self.assertTrue(tk.allclose(dex, tk.exp(tk.array(1.0))))
        self.assertTrue(tk.allclose(ex, dex))
        ex = tk.vmap(my_exp)(tk.ones(10))
        self.assertTrue(tk.allclose(ex, tk.exp(tk.ones(10))))

        # Ensure that the vjp is being overriden but everything else still
        # works.
        @my_exp.vjp
        def my_exp_vjp(x, dx, ex):
            return tk.ones_like(x) * 42

        dy = tk.grad(my_exp)(tk.array(1.0))
        self.assertTrue(tk.allclose(dy, tk.array(42.0)))
        (ex,), (dex,) = tk.jvp(my_exp, [tk.array(1.0)], [tk.array(1.0)])
        self.assertTrue(tk.allclose(dex, tk.exp(tk.array(1.0))))
        self.assertTrue(tk.allclose(ex, dex))
        ex = tk.vmap(my_exp)(tk.ones(10))
        self.assertTrue(tk.allclose(ex, tk.exp(tk.ones(10))))

        # Ensure that setting the jvp and vmap also works.
        @my_exp.jvp
        def my_exp_jvp(x, dx):
            return tk.ones_like(x) * 7 * dx

        @my_exp.vmap
        def my_exp_vmap(x, axis):
            return tk.ones_like(x) * 3, axis

        dy = tk.grad(my_exp)(tk.array(1.0))
        self.assertTrue(tk.allclose(dy, tk.array(42.0)))
        (ex,), (dex,) = tk.jvp(my_exp, [tk.array(1.0)], [tk.array(1.0)])
        self.assertTrue(tk.allclose(dex, tk.array(7.0)))
        self.assertTrue(tk.allclose(ex, tk.exp(tk.array(1.0))))
        ex = tk.vmap(my_exp)(tk.ones(10))
        self.assertTrue(tk.allclose(ex, 3 * tk.ones(10)))

        # Test pytrees
        @tk.custom_function
        def my_double(params):
            return {"out": 2 * params["x"] * params["y"]}

        dy = tk.grad(lambda p: my_double(p)["out"].sum())(
            {"x": tk.ones(2), "y": tk.ones(2)}
        )
        self.assertTrue(tk.allclose(dy["x"], tk.ones(2) * 2))
        self.assertTrue(tk.allclose(dy["y"], tk.ones(2) * 2))

        @my_double.vjp
        def random_grads(primals, cotangents, outputs):
            return {"x": tk.zeros_like(primals["x"]), "y": tk.ones_like(primals["y"])}

        dy = tk.grad(lambda p: my_double(p)["out"].sum())(
            {"x": tk.ones(2), "y": tk.ones(2)}
        )
        self.assertTrue(tk.allclose(dy["x"], tk.zeros(2)))
        self.assertTrue(tk.allclose(dy["y"], tk.ones(2)))

        def outer_f(a, b):
            return my_double({"x": a, "y": b})["out"]

        inputs = [tk.random.normal(shape=(2,)) for i in range(2)]
        tans = [tk.random.normal(shape=(2,)) for i in range(2)]
        out1, dout1 = tk.jvp(outer_f, inputs, tans)

        @my_double.jvp
        def random_grads(primals, tangents):
            return {
                "out": 2 * primals["x"] * tangents["y"]
                + 2 * primals["y"] * tangents["x"]
                + 1
            }

        out2, dout2 = tk.jvp(outer_f, inputs, tans)
        self.assertTrue(tk.allclose(out1[0], out2[0]))
        self.assertTrue(tk.allclose(dout1[0] + 1, dout2[0]))

    def test_complex_vjps(self):
        def fun(x):
            return (2.0 * tk.real(x)).sum()

        x = tk.array([0.0 + 1j, 1.0 + 0.0j, 0.5 + 0.5j])
        dfdx = tk.grad(fun)(x)
        self.assertTrue(tk.allclose(dfdx, 2 * tk.ones_like(x)))

        def fun(x):
            return (2.0 * tk.imag(x)).sum()

        x = tk.array([0.0 + 1j, 1.0 + 0.0j, 0.5 + 0.5j])
        dfdx = tk.grad(fun)(x)
        self.assertTrue(tk.allclose(dfdx, 2j * tk.ones_like(x)))

    def test_flatten_unflatten_vjps(self):
        def fun(x):
            y = tk.unflatten(x, 0, (2, 2))
            return y.sum()

        x = tk.zeros((4, 8))
        self.assertEqual(tk.grad(fun)(x).shape, (4, 8))

        def fun(x):
            y = tk.flatten(x, 0, 2)
            return y.sum()

        x = tk.zeros((2, 4, 8))
        self.assertEqual(tk.grad(fun)(x).shape, (2, 4, 8))

    def test_concatenate_vjps(self):
        def fun(x, y):
            return tk.concatenate([x, y])

        x = tk.array([1, 2, 3], tk.float32)
        y = tk.array([1, 2, 3], tk.float16)
        grads = tk.vjp(fun, (x, y), (tk.ones((6,)),))[1]
        self.assertTrue(tk.allclose(grads[0], tk.ones(3)))
        self.assertTrue(tk.allclose(grads[1], tk.ones(3)))
        self.assertEqual(grads[0].dtype, tk.float32)
        self.assertEqual(grads[1].dtype, tk.float16)

    def test_matmul_jvps(self):
        a = tk.random.uniform(shape=(4, 4))
        b = tk.random.uniform(shape=(4, 4))
        c = tk.random.uniform(shape=(4, 4))
        d = tk.random.uniform(shape=(4, 4))

        _, tangent = tk.jvp(lambda a: a @ b, (a,), (c,))
        self.assertTrue(tk.allclose(tangent[0], c @ b))

        _, tangent = tk.jvp(lambda b: a @ b, (b,), (d,))
        self.assertTrue(tk.allclose(tangent[0], a @ d))

        _, tangent = tk.jvp(lambda a, b: a @ b, (a, b), (c, d))
        self.assertTrue(tk.allclose(tangent[0], a @ d + c @ b))

        x = tk.random.uniform(shape=(4, 4))
        y = tk.random.uniform(shape=(4, 4))
        z = tk.random.uniform(shape=(4, 4))

        _, (tangent,) = tk.jvp(lambda a, b, c: a @ b + c, (a, b, c), (x, y, z))
        _, (expected,) = tk.jvp(lambda a, b, c: tk.addmm(c, a, b), (a, b, c), (x, y, z))
        self.assertTrue(tk.allclose(tangent, expected))

        _, (tangent,) = tk.jvp(lambda a, c: a @ b + c, (a, c), (x, z))
        _, (expected,) = tk.jvp(lambda a, c: tk.addmm(c, a, b), (a, c), (x, z))
        self.assertTrue(tk.allclose(tangent, expected))

        _, (tangent,) = tk.jvp(lambda b, c: a @ b + c, (b, c), (y, z))
        _, (expected,) = tk.jvp(lambda b, c: tk.addmm(c, a, b), (b, c), (y, z))
        self.assertTrue(tk.allclose(tangent, expected))

        _, (tangent,) = tk.jvp(lambda c: a @ b + c, (c,), (z,))
        _, (expected,) = tk.jvp(lambda c: tk.addmm(c, a, b), (c,), (z,))
        self.assertTrue(tk.allclose(tangent, expected))

    def test_put_along_axis_grads(self):
        a = tk.zeros((5, 1))
        b = tk.ones((2, 1))

        def fun(a, b):
            idx = tk.array([[0], [3]])
            return tk.put_along_axis(a, idx, b, axis=0)

        # Test VJP
        cotan = tk.full((5, 1), 2.0)
        _, (da, db) = tk.vjp(fun, (a, b), (cotan,))
        expected_da = tk.array([0.0, 2.0, 2.0, 0.0, 2.0])[:, None]
        expected_db = tk.array([2.0, 2.0])[:, None]
        self.assertTrue(tk.allclose(expected_da, da))
        self.assertTrue(tk.allclose(expected_db, db))

        # Test JVP
        tan_a = tk.full((5, 1), 2.0)
        tan_b = tk.full((2, 1), 3.0)
        _, (jout,) = tk.jvp(fun, (a, b), (tan_a, tan_b))
        expected = tk.array([3.0, 2.0, 2.0, 3.0, 2.0])[:, None]
        self.assertTrue(tk.allclose(expected, jout))

        def fun(a):
            idx = tk.array([[0], [3]])
            return tk.put_along_axis(a, idx, b, axis=0)

        _, (jout,) = tk.jvp(fun, (a,), (tan_a,))
        expected = tk.array([0.0, 2.0, 2.0, 0.0, 2.0])[:, None]
        self.assertTrue(tk.allclose(expected, jout))

    def test_slice_grads(self):
        # Slice
        def fun(a):
            return a[5:-6:-1]

        a = tk.ones(shape=(5,))
        cotan = tk.random.uniform(shape=(5,))
        _, (grad,) = tk.vjp(fun, (a,), (cotan,))
        self.assertTrue(tk.allclose(grad, cotan[::-1]))

        tan = tk.random.uniform(shape=(5,))
        tk.eval(tan)
        _, (grad,) = tk.jvp(fun, (a,), (tan,))
        self.assertTrue(tk.allclose(grad, tan[::-1]))

        # Slice update
        def fun(a, b):
            a[4:-5:-2] = b
            return a

        a = tk.ones(shape=(4,))
        b = tk.zeros(shape=(2,))

        cotan = tk.random.uniform(shape=(4,))
        _, (grad_a, grad_b) = tk.vjp(fun, (a, b), (cotan,))
        expected_a = tk.array(cotan)
        expected_a[1::2] = 0.0
        self.assertTrue(tk.allclose(grad_a, expected_a))
        self.assertTrue(tk.allclose(grad_b, cotan[4:-5:-2]))

        tan_a = tk.random.uniform(shape=(4,))
        tan_b = tk.random.uniform(shape=(2,))
        _, (grad,) = tk.jvp(fun, (a, b), (tan_a, tan_b))
        expected = tan_a
        expected[4:-5:-2] = tan_b
        self.assertTrue(tk.allclose(grad, expected))

    def test_leaks(self):
        for transform in [
            tk.grad,
            tk.value_and_grad,
            tk.custom_function,
            tk.checkpoint,
        ]:
            tk.synchronize()
            gc.collect()
            mem_pre = tk.get_active_memory()

            def outer():
                d = {}

                def f(x):
                    return d["x"]

                d["f"] = transform(f)
                d["x"] = tk.array([0] * 1000)

            for _ in range(5):
                outer()
                gc.collect()
            mem_post = tk.get_active_memory()
            self.assertEqual(mem_pre, mem_post)

    def test_grad_with_copies(self):
        a = tk.array(2.0)
        arrays = [a, a, a]

        def fun(arrays):
            return arrays[0] + arrays[2]

        grads = tk.grad(fun)(arrays)
        self.assertEqual(grads[0].item(), 1.0)
        self.assertEqual(grads[2].item(), 1.0)

    def test_grad_ids_pre_post(self):
        def fun(arrs):
            return arrs[0]

        arrs = [tk.array(1.0)]
        arr = arrs[0]
        tk.grad(fun)(arrs)
        self.assertEqual(id(arr), id(arrs[0]))

        def fun(arrs):
            arrs[1] = sum(arrs)
            return arrs[1]

        arrs = [tk.array(1.0), tk.array(1.0), tk.array(1.0)]
        a_0, a_1, a_2 = arrs

        tk.grad(fun)(arrs)
        self.assertEqual(id(a_0), id(arrs[0]))
        self.assertNotEqual(id(a_1), id(arrs[1]))
        self.assertEqual(id(a_2), id(arrs[2]))

    def test_grad_with_inplace_update(self):
        def loss_fn(model):
            model[1] = tk.array(2.0)
            return model[0]

        model = [
            tk.array(0.0),
            tk.array(1.0),
        ]

        grad_fn = tk.grad(loss_fn)
        grad_fn(model)
        self.assertEqual(model[1].item(), 2.0)

    def test_autograd_types(self):
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

        def loss_fn(x):
            out = transform(x)
            return out.a.sum() + out.b.sum()

        def loss_fn_tuple(x):
            out = transform_tuple(x)
            return out[0].sum() + out[1].sum()

        def loss_fn_vector(x):
            out = transform_vector(x)
            return out[0].sum() + out[1].sum()

        x_batch = State(tk.array([1, 2, 3]), tk.array([4, 5, 6]))
        grads = tk.grad(loss_fn)(x_batch)
        self.assertTrue(isinstance(grads, State))
        self.assertTrue(tk.array_equal(grads.a, tk.ones(3)))
        self.assertTrue(tk.array_equal(grads.b, tk.ones(3) * 10))

        x_batch_tuple = (tk.array([1, 2, 3]), tk.array([4, 5, 6]))
        grads = tk.grad(loss_fn_tuple)(x_batch_tuple)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(tk.array_equal(grads[0], tk.ones(3)))
        self.assertTrue(tk.array_equal(grads[1], tk.ones(3) * 10))

        x_batch_vector = Vector([tk.array([1, 2, 3]), tk.array([4, 5, 6])])
        grads = tk.grad(loss_fn_vector)(x_batch_vector)
        self.assertTrue(isinstance(grads, Vector))
        self.assertTrue(tk.array_equal(grads[0], tk.ones(3)))
        self.assertTrue(tk.array_equal(grads[1], tk.ones(3) * 10))

    def test_reduce_jvp(self):
        a = tk.arange(4)
        b = tk.array([3, 2, 1, 0])

        out, jout = tk.jvp(tk.sum, primals=(a,), tangents=(b,))
        self.assertEqual(jout[0].item(), 6)

        out, jout = tk.jvp(tk.prod, primals=(a,), tangents=(b,))
        self.assertEqual(jout[0].item(), 18)

        out, jout = tk.jvp(tk.min, primals=(a,), tangents=(b,))
        self.assertEqual(jout[0].item(), 3)

        out, jout = tk.jvp(tk.max, primals=(a,), tangents=(b,))
        self.assertEqual(jout[0].item(), 0)

    def test_complex_prod_vjp(self):
        def prod(x):
            return x.prod(axis=0)

        primal = tk.random.normal((2, 20), dtype=tk.complex64)
        cotangent = tk.random.normal((20,), dtype=tk.complex64)

        _, vjps = tk.vjp(prod, [primal], [cotangent])

        expected = tk.stack(
            [tk.conj(primal[1]) * cotangent, tk.conj(primal[0]) * cotangent]
        )

        # Check against hand-computed vjps
        self.assertTrue(tk.array_equal(vjps[0], expected))

        # Ensure that prod agrees with multiply for complex values
        _, vjps_multiply = tk.vjp(tk.multiply, [primal[0], primal[1]], [cotangent])

        self.assertTrue(tk.array_equal(tk.stack(vjps_multiply), vjps[0]))

    def test_complex_exp_vjp(self):
        primal = tk.random.normal((3, 4, 5), dtype=tk.complex64)
        cotangent = tk.random.normal(
            (
                3,
                4,
                5,
            ),
            dtype=tk.complex64,
        )

        _, vjps = tk.vjp(tk.exp, [primal], [cotangent])

        expected = cotangent * tk.conj(tk.exp(primal))

        # Check against hand-computed vjps
        self.assertTrue(tk.allclose(vjps[0], expected))

    def test_complex_log_vjp(self):
        primal = tk.random.normal((3, 4, 5), dtype=tk.complex64)
        cotangent = tk.random.normal(
            (
                3,
                4,
                5,
            ),
            dtype=tk.complex64,
        )

        # guard against values too close to the origin
        primal = tk.where(abs(primal) < 1e-3, 1e-3, primal)

        _, vjps = tk.vjp(tk.log, [primal], [cotangent])

        expected = cotangent * tk.conj(1 / primal)

        # Check against hand-computed vjps
        self.assertTrue(tk.allclose(vjps[0], expected))

    def test_complex_unary_vjps(self):
        # For a holomorphic f the vjp is cotangent * conj(f'(z)); these ops used
        # to delegate to their jvp and drop the conjugate for complex inputs.
        tk.random.seed(0)
        z = tk.random.normal((3, 4, 5), dtype=tk.complex64)
        cotangent = tk.random.normal((3, 4, 5), dtype=tk.complex64)
        z = tk.where(abs(z) < 1e-3, 1e-3 + 0j, z)

        ops = {
            tk.square: lambda x: 2 * x,
            tk.sin: tk.cos,
            tk.sinh: tk.cosh,
            tk.cosh: tk.sinh,
            tk.tan: lambda x: 1 / tk.cos(x) ** 2,
            tk.tanh: lambda x: 1 - tk.tanh(x) ** 2,
            tk.log1p: lambda x: 1 / (1 + x),
        }
        for fn, deriv in ops.items():
            _, (vjp,) = tk.vjp(fn, [z], [cotangent])
            expected = cotangent * tk.conj(deriv(z))
            self.assertTrue(tk.allclose(vjp, expected, atol=1e-5), msg=str(fn))

    def test_complex_abs_grad(self):
        tk.random.seed(0)
        primal = tk.random.normal((3, 4, 5), dtype=tk.complex64)
        # guard against values too close to the origin where |z| is not smooth
        primal = tk.where(abs(primal) < 1e-3, 1e-3 + 0j, primal)

        # |z| is real-valued, so its jvp is real:
        #   d|z| = Re(conj(z) * t) / |z|
        tangent = tk.random.normal(primal.shape, dtype=tk.complex64)
        _, (jvp,) = tk.jvp(tk.abs, [primal], [tangent])
        expected = tk.real(tk.conj(primal) * tangent) / tk.abs(primal)
        self.assertEqual(jvp.dtype, tk.float32)
        self.assertTrue(tk.allclose(jvp, expected, atol=1e-5))

        # The vjp's real and imaginary parts are the gradients w.r.t. Re(z) and
        # Im(z); for a real cotangent this is cotangent * sign(z).
        cotangent = tk.random.normal(primal.shape)
        _, (vjp,) = tk.vjp(tk.abs, [primal], [cotangent])
        self.assertTrue(
            tk.allclose(vjp, cotangent * (primal / tk.abs(primal)), atol=1e-5)
        )

        # Real inputs are unaffected.
        x = tk.random.normal((10,))
        t = tk.random.normal((10,))
        _, (jvp,) = tk.jvp(tk.abs, [x], [t])
        self.assertTrue(tk.allclose(jvp, tk.sign(x) * t))

    def test_second_order_permutation_ops(self):
        # The permutation these ops apply is locally constant in the input, so
        # the indices must not carry a gradient. Otherwise differentiating the
        # vjp a second time fails with "Cannot calculate VJP with respect to
        # indices".
        def hvp(f, x, v):
            return tk.grad(lambda a: tk.sum(tk.grad(f)(a) * v))(x)

        def numerical_hvp(f, x, v, eps=1e-3):
            # The values below are well separated, so the permutation does not
            # change over this step and the difference is exact enough.
            return (tk.grad(f)(x + eps * v) - tk.grad(f)(x - eps * v)) / (2 * eps)

        x = tk.array([3.0, 1.0, 2.0, 5.0])
        v = tk.array([1.0, -2.0, 0.5, 1.5])

        for fn in (
            lambda a: tk.sort(a),
            lambda a: tk.partition(a, 2),
            lambda a: tk.topk(a, 2),
            lambda a: tk.cummax(a, axis=0),
            lambda a: tk.cummin(a, axis=0),
            lambda a: tk.cummax(a, axis=0, reverse=True),
            lambda a: tk.cummax(a, axis=0, inclusive=False),
            lambda a: tk.cummin(a, axis=0, reverse=True, inclusive=False),
        ):
            f = lambda a: tk.sum(fn(a) ** 2)
            self.assertTrue(
                tk.allclose(hvp(f, x, v), numerical_hvp(f, x, v), atol=1e-3)
            )

        # A non-trailing axis
        y = tk.array([[3.0, 1.0], [2.0, 5.0]])
        w = tk.array([[1.0, -2.0], [0.5, 1.5]])
        for fn in (
            lambda a: tk.sort(a, axis=0),
            lambda a: tk.partition(a, 1, axis=0),
            lambda a: tk.cummax(a, axis=0),
        ):
            f = lambda a: tk.sum(fn(a) ** 2)
            self.assertTrue(
                tk.allclose(hvp(f, y, w), numerical_hvp(f, y, w), atol=1e-3)
            )


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
