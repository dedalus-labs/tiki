# Copyright © 2023-2024 Apple Inc.

import gc
import unittest

import tiki as tk
import tiki_tests


class TestVmap(tiki_tests.TIKITestCase):
    def test_basics(self):
        # Can't vmap over scalars
        with self.assertRaises(ValueError):
            tk.vmap(tk.exp)(tk.array(1.0))

        # Invalid input
        with self.assertRaises(ValueError):
            tk.vmap(tk.exp)("hello")

        # Invalid axes
        with self.assertRaises(ValueError):
            tk.vmap(tk.exp, in_axes="hello")(tk.array([0, 1]))

        with self.assertRaises(ValueError):
            tk.vmap(tk.exp, in_axes=2)(tk.array([0, 1]))

        with self.assertRaises(ValueError):
            tk.vmap(tk.exp, out_axes="hello")(tk.array([0, 1]))

        with self.assertRaises(ValueError):
            tk.vmap(tk.exp, out_axes=2)(tk.array([0, 1]))

    def test_unary(self):
        ops = [
            "abs",
            "cos",
            "erf",
            "erfinv",
            "exp",
            "log",
            "log1p",
            "log2",
            "log10",
            "logical_not",
            "negative",
            "reciprocal",
            "rsqrt",
            "sigmoid",
            "sign",
            "sin",
            "sqrt",
            "square",
            "degrees",
            "radians",
        ]
        for opname in ops:
            with self.subTest(op=opname):
                op = getattr(tk, opname)
                x = tk.arange(5)
                y = tk.vmap(op)(x)
                self.assertTrue(tk.array_equal(y, op(x), equal_nan=True))

                x = tk.arange(8).reshape(2, 4)
                y = tk.vmap(op)(x)
                self.assertTrue(tk.array_equal(y, op(x), equal_nan=True))

                y = tk.vmap(op, in_axes=1, out_axes=1)(x)
                self.assertTrue(tk.array_equal(y, op(x), equal_nan=True))

    def test_binary(self):
        ops = [
            "add",
            "divide",
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logaddexp",
            "maximum",
            "minimum",
            "multiply",
            "power",
            "subtract",
            "logical_or",
            "logical_and",
        ]
        for opname in ops:
            with self.subTest(op=opname):
                op = getattr(tk, opname)
                x = tk.random.uniform(shape=(5,))
                y = tk.random.uniform(shape=(5,))
                out = tk.vmap(op)(x, y)
                self.assertTrue(tk.array_equal(out, op(x, y)))

                x = tk.random.uniform(shape=(2, 4))
                y = tk.random.uniform(shape=(2, 4))
                out = tk.vmap(op)(x, y)
                self.assertTrue(tk.array_equal(out, op(x, y)))

                out = tk.vmap(op, in_axes=(0, 0), out_axes=0)(x, y)
                self.assertTrue(tk.array_equal(out, op(x, y)))

                y = tk.random.uniform(shape=(4, 2))
                out = tk.vmap(op, in_axes=(0, 1), out_axes=0)(x, y)
                self.assertTrue(tk.array_equal(out, op(x, y.T)))

                out = tk.vmap(op, in_axes=(0, 1), out_axes=1)(x, y)
                self.assertTrue(tk.array_equal(out, op(x, y.T).T))

    def test_tree(self):
        def my_fun(tree):
            return (tree["a"] + tree["b"][0]) * tree["b"][1]

        tree = {
            "a": tk.random.uniform(shape=(2, 4)),
            "b": (
                tk.random.uniform(shape=(2, 4)),
                tk.random.uniform(shape=(2, 4)),
            ),
        }
        out = tk.vmap(my_fun)(tree)
        expected = my_fun(tree)
        self.assertTrue(tk.array_equal(out, my_fun(tree)))

        with self.assertRaises(ValueError):
            tk.vmap(my_fun, in_axes={"a": 0, "b": ((0, 0), 0)}, out_axes=0)(tree)

        out = tk.vmap(my_fun, in_axes={"a": 0, "b": 0}, out_axes=0)(tree)
        self.assertTrue(tk.array_equal(out, my_fun(tree)))

        out = tk.vmap(my_fun, in_axes={"a": 0, "b": (0, 0)}, out_axes=0)(tree)
        self.assertTrue(tk.array_equal(out, my_fun(tree)))

        tree = {
            "a": tk.random.uniform(shape=(2, 4)),
            "b": (
                tk.random.uniform(shape=(4, 2)),
                tk.random.uniform(shape=(4, 2)),
            ),
        }
        out = tk.vmap(my_fun, in_axes={"a": 0, "b": (1, 1)}, out_axes=0)(tree)
        expected = (tree["a"] + tree["b"][0].T) * tree["b"][1].T
        self.assertTrue(tk.array_equal(out, expected))

        def my_fun(x, y):
            return {"a": x + y, "b": x * y}

        x = tk.random.uniform(shape=(2, 4))
        y = tk.random.uniform(shape=(2, 4))
        out = tk.vmap(my_fun, in_axes=0, out_axes=0)(x, y)
        expected = my_fun(x, y)
        self.assertTrue(tk.array_equal(out["a"], expected["a"]))
        self.assertTrue(tk.array_equal(out["b"], expected["b"]))

        with self.assertRaises(ValueError):
            tk.vmap(my_fun, in_axes=0, out_axes=(0, 1))(x, y)

        with self.assertRaises(ValueError):
            tk.vmap(my_fun, in_axes=0, out_axes={"a": 0, "c": 1})(x, y)

        out = tk.vmap(my_fun, in_axes=0, out_axes={"a": 1, "b": 0})(x, y)
        expected = my_fun(x, y)
        self.assertTrue(tk.array_equal(out["a"].T, expected["a"]))
        self.assertTrue(tk.array_equal(out["b"], expected["b"]))

    def test_vmap_indexing(self):
        x = tk.arange(16).reshape(2, 2, 2, 2)
        inds = tk.array([[0, 1, 0], [1, 1, 0]])

        out = tk.vmap(lambda x, y: x[y], in_axes=(0, 0))(x, inds)
        expected = tk.array(
            [
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
                [[[12, 13], [14, 15]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
            ]
        )
        self.assertTrue(tk.array_equal(out, expected))

        out = tk.vmap(lambda x, y: x[y], in_axes=(0, None))(x, inds)
        expected = tk.array(
            [
                [
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
                    [[[4, 5], [6, 7]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
                ],
                [
                    [[[8, 9], [10, 11]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
                    [[[12, 13], [14, 15]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
                ],
            ]
        )
        self.assertTrue(tk.array_equal(out, expected))

        out = tk.vmap(lambda x, y: x[y], in_axes=(None, 0))(x, inds)
        expected = tk.array(
            [
                [
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                    [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                ],
                [
                    [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
                    [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                ],
            ]
        )
        self.assertTrue(tk.array_equal(out, expected))

        inds2 = tk.array([[0, 1, 0], [0, 1, 0]])
        out = tk.vmap(lambda x, y, z: x[y, z], in_axes=(None, 0, 0))(x, inds, inds2)
        expected = tk.array(
            [
                [[[0, 1], [2, 3]], [[12, 13], [14, 15]], [[0, 1], [2, 3]]],
                [[[8, 9], [10, 11]], [[12, 13], [14, 15]], [[0, 1], [2, 3]]],
            ]
        )
        self.assertTrue(tk.array_equal(out, expected))

    def test_vmap_reduce(self):
        a = tk.ones((5, 5), tk.int32)
        out = tk.vmap(lambda x: x.sum())(a)
        self.assertTrue(tk.array_equal(out, tk.full((5,), 5)))

        out = tk.vmap(lambda x: x.sum(keepdims=True))(a)
        self.assertTrue(tk.array_equal(out, tk.full((5, 1), 5)))

        out = tk.vmap(lambda x: x.sum(axis=0))(a)
        self.assertTrue(tk.array_equal(out, tk.full((5,), 5)))

        a = tk.ones((5, 3, 2), tk.int32)
        out = tk.vmap(lambda x: x.sum(axis=(0, 1)))(a)
        self.assertTrue(tk.array_equal(out, tk.full((5,), 6)))

        a = tk.ones((5, 3, 2), tk.int32)
        out = tk.vmap(lambda x: x.sum(axis=(0, 1)), in_axes=(1,))(a)
        self.assertTrue(tk.array_equal(out, tk.full((3,), 10)))

        a = tk.ones((5, 3, 2), tk.int32)
        out = tk.vmap(lambda x: x.sum(axis=(0, 1)), in_axes=(2,))(a)
        self.assertTrue(tk.array_equal(out, tk.full((2,), 15)))

    def test_vmap_argreduce(self):
        a = tk.array([[1, 2, 3], [2, 3, 1]])
        out = tk.vmap(lambda x: tk.argmin(x))(a)
        expected = tk.array([0, 2])
        self.assertTrue(tk.array_equal(out, expected))

        out = tk.vmap(lambda x: tk.argmax(x))(a)
        expected = tk.array([2, 1])
        self.assertTrue(tk.array_equal(out, expected))

    def _unstack(self, x, axis):
        return [s.squeeze(axis) for s in tk.split(x, x.shape[axis], axis=axis)]

    def test_vmap_partition(self):
        # Distinct values so each lane has a single valid kth element
        a = tk.random.permutation(2 * 3 * 4).reshape(2, 3, 4).astype(tk.float32)

        for in_axis in (0, 1, 2):
            slices = self._unstack(a, in_axis)
            # Axis of the batched output that the inner axis maps onto
            out_axes_map = [d for d in range(a.ndim) if d != in_axis]
            for axis in (0, 1, -1):
                oaxis = out_axes_map[axis if axis >= 0 else axis + 2]
                for kth in range(slices[0].shape[axis]):
                    expected = tk.stack(
                        [tk.partition(x, kth, axis=axis) for x in slices],
                        axis=in_axis,
                    )
                    pivot = tk.take(expected, tk.array([kth]), axis=oaxis)

                    out = tk.vmap(
                        lambda x: tk.partition(x, kth, axis=axis),
                        in_axes=in_axis,
                        out_axes=in_axis,
                    )(a)
                    self.assertEqual(out.shape, expected.shape)
                    # partition only pins the kth element; the two sides are
                    # an arbitrary permutation, so compare against the sorted
                    # input rather than element-wise.
                    self.assertTrue(
                        tk.array_equal(tk.sort(out, axis=oaxis), tk.sort(a, axis=oaxis))
                    )
                    self.assertTrue(
                        tk.array_equal(tk.take(out, tk.array([kth]), axis=oaxis), pivot)
                    )

                    idx = tk.vmap(
                        lambda x: tk.argpartition(x, kth, axis=axis),
                        in_axes=in_axis,
                        out_axes=in_axis,
                    )(a)
                    self.assertEqual(idx.shape, expected.shape)
                    gathered = tk.take_along_axis(a, idx, axis=oaxis)
                    self.assertTrue(
                        tk.array_equal(
                            tk.sort(gathered, axis=oaxis), tk.sort(a, axis=oaxis)
                        )
                    )
                    self.assertTrue(
                        tk.array_equal(
                            tk.take(gathered, tk.array([kth]), axis=oaxis), pivot
                        )
                    )

    def test_vmap_topk(self):
        a = tk.random.permutation(2 * 3 * 4).reshape(2, 3, 4).astype(tk.float32)

        for in_axis in (0, 1, 2):
            slices = self._unstack(a, in_axis)
            out_axes_map = [d for d in range(a.ndim) if d != in_axis]
            for axis in (0, 1, -1):
                oaxis = out_axes_map[axis if axis >= 0 else axis + 2]
                for k in range(1, slices[0].shape[axis] + 1):
                    out = tk.vmap(
                        lambda x: tk.topk(x, k, axis=axis),
                        in_axes=in_axis,
                        out_axes=in_axis,
                    )(a)
                    expected = tk.stack(
                        [tk.topk(x, k, axis=axis) for x in slices], axis=in_axis
                    )
                    self.assertEqual(out.shape, expected.shape)
                    # topk does not promise an order within the k elements
                    self.assertTrue(
                        tk.array_equal(
                            tk.sort(out, axis=oaxis), tk.sort(expected, axis=oaxis)
                        )
                    )

    def test_vmap_mean(self):
        a = tk.arange(8).reshape(2, 4)
        out = tk.vmap(tk.mean)(a)
        expected = tk.mean(a, axis=1)
        self.assertTrue(tk.allclose(out, expected))

        a = tk.arange(16).reshape(2, 2, 4)
        out = tk.vmap(tk.vmap(tk.mean))(a)
        expected = tk.mean(a, axis=2)
        self.assertTrue(tk.allclose(out, expected))

    def test_mismatch_input_sizes(self):
        a = tk.ones((10, 1))
        b = tk.ones((1, 1, 1, 5))

        with self.assertRaises(ValueError):
            out = tk.vmap(lambda x, y: x + y)(a, b)

        b = tk.ones((10, 5))
        with self.assertRaises(ValueError):
            out = tk.vmap(lambda x, y: x + y, in_axes=(0, 1))(a, b)

    def test_vmap_matmul(self):
        a = tk.random.uniform(shape=(2, 3, 4))
        b = tk.random.uniform(shape=(4, 3))

        # matmul
        out = tk.vmap(tk.matmul, in_axes=(0, None))(a, b)
        self.assertTrue(tk.allclose(out, a @ b))

        # addmm
        c = tk.random.uniform(shape=(3,))
        out = tk.vmap(tk.addmm, in_axes=(None, 0, None))(c, a, b)
        self.assertTrue(tk.allclose(out, tk.addmm(c, a, b)))

        b = tk.random.uniform(shape=(4, 2))

        # matmul
        out = tk.vmap(tk.matmul, in_axes=(1, None), out_axes=(1,))(a, b)
        expected = tk.moveaxis(tk.moveaxis(a, 1, 0) @ b, 0, 1)
        self.assertTrue(tk.allclose(out, expected))

        # addmm
        c = tk.random.uniform(shape=(2,))
        out = tk.vmap(tk.addmm, in_axes=(None, 1, None))(c, a, b)
        self.assertTrue(tk.allclose(out, tk.addmm(c, tk.moveaxis(a, 1, 0), b)))

        a = tk.random.uniform(shape=(2, 3, 4))
        b = tk.random.uniform(shape=(4, 2, 3))

        # matmul
        out = tk.vmap(tk.matmul, in_axes=(0, 1))(a, b)
        expected = a @ tk.moveaxis(b, 1, 0)
        self.assertTrue(tk.allclose(out, expected))

        # addmm
        c = tk.random.uniform(shape=(3, 3, 2))
        out = tk.vmap(tk.addmm, in_axes=(2, 0, 1))(c, a, b)
        expected = tk.addmm(tk.moveaxis(c, 2, 0), a, tk.moveaxis(b, 1, 0))
        self.assertTrue(tk.allclose(out, expected))

    def test_vmap_svd(self):
        a = tk.random.uniform(shape=(3, 4, 2))

        cpu_svd_full = lambda x: tk.linalg.svd(x, compute_uv=True, stream=tk.cpu)
        cpu_svd_singular = lambda x: tk.linalg.svd(x, compute_uv=False, stream=tk.cpu)

        # Vmap over the first axis (this is already supported natively by the primitive).
        Us, Ss, Vts = tk.vmap(cpu_svd_full, in_axes=(0,))(a)
        self.assertEqual(Us.shape, (a.shape[0], a.shape[1], a.shape[1]))
        self.assertEqual(Ss.shape, (a.shape[0], a.shape[2]))
        self.assertEqual(Vts.shape, (a.shape[0], a.shape[2], a.shape[2]))

        Sv = tk.vmap(cpu_svd_singular, in_axes=(0,))(a)
        self.assertEqual(Sv.shape, (a.shape[0], a.shape[2]))

        for i in range(a.shape[0]):
            M = a[i]
            U, S, Vt = Us[i], Ss[i], Vts[i]
            self.assertTrue(
                tk.allclose(U[:, : len(S)] @ tk.diag(S) @ Vt, M, rtol=1e-5, atol=1e-7)
            )
            self.assertTrue(
                tk.allclose(
                    tk.linalg.norm(Sv[i]),
                    tk.linalg.norm(M, ord="fro"),
                    rtol=1e-5,
                    atol=1e-7,
                )
            )

        # Vmap over the second axis.
        Us, Ss, Vts = tk.vmap(cpu_svd_full, in_axes=(1,))(a)
        self.assertEqual(Us.shape, (a.shape[1], a.shape[0], a.shape[0]))
        self.assertEqual(Ss.shape, (a.shape[1], a.shape[2]))
        self.assertEqual(Vts.shape, (a.shape[1], a.shape[2], a.shape[2]))

        Sv = tk.vmap(cpu_svd_singular, in_axes=(1,))(a)
        self.assertEqual(Sv.shape, (a.shape[1], a.shape[2]))

        for i in range(a.shape[1]):
            M = a[:, i, :]
            U, S, Vt = Us[i], Ss[i], Vts[i]
            self.assertTrue(
                tk.allclose(U[:, : len(S)] @ tk.diag(S) @ Vt, M, rtol=1e-5, atol=1e-7)
            )
            self.assertTrue(
                tk.allclose(
                    tk.linalg.norm(Sv[i]),
                    tk.linalg.norm(M, ord="fro"),
                    rtol=1e-5,
                    atol=1e-7,
                )
            )

    def test_vmap_inverse(self):
        tk.random.seed(42)
        a = tk.random.uniform(shape=(3, 4, 4))

        cpu_inv = lambda x: tk.linalg.inv(x, stream=tk.cpu)

        # Vmap over the first axis (this is already supported natively by the primitive).
        invs = tk.vmap(cpu_inv, in_axes=(0,))(a)

        for i in range(a.shape[0]):
            self.assertTrue(
                tk.allclose(a[i] @ invs[i], tk.eye(a.shape[1]), rtol=1e-4, atol=1e-5)
            )

        a = tk.random.uniform(shape=(4, 3, 4))

        # Without vmapping, each input matrix is not square.
        with self.assertRaises(ValueError):
            tk.eval(cpu_inv(a))

        # Vmap over the second axis.
        invs = tk.vmap(cpu_inv, in_axes=(1,))(a)

        for i in range(a.shape[1]):
            self.assertTrue(
                tk.allclose(
                    a[:, i, :] @ invs[i], tk.eye(a.shape[0]), rtol=1e-4, atol=1e-5
                )
            )

    def test_vmap_gather(self):
        def gather(a, idx):
            return a[idx]

        a = tk.array([[1, 2], [3, 4]])
        idx = tk.array(0)
        out = tk.vmap(gather, (0, None))(a, idx)
        self.assertTrue(tk.array_equal(out, tk.array([1, 3])))

        out = tk.vmap(gather, (1, None))(a, idx)
        self.assertTrue(tk.array_equal(out, tk.array([1, 2])))

        idx = tk.array([0, 1])
        out = tk.vmap(gather, (0, 0))(a, idx)
        self.assertTrue(tk.array_equal(out, tk.array([1, 4])))

        a = tk.ones((2, 3, 4))
        idx = tk.zeros(4, tk.int32)
        out = tk.vmap(gather, (2, 0))(a, idx)
        self.assertEqual(out.shape, (4, 3))

        f = tk.vmap(gather, (0, None))
        f = tk.vmap(gather, (0, 0))
        out = f(tk.ones((2, 3, 4)), tk.zeros(2, dtype=tk.int32))
        self.assertEqual(out.shape, (2, 4))

        def gather(a, idxa, idxb):
            return a[idxa, idxb]

        a = tk.ones((2, 3, 4))
        idxa = tk.zeros((2, 3), tk.int32)
        idxb = tk.zeros(3, tk.int32)
        out = tk.vmap(gather, (0, 0, None))(a, idxa, idxb)
        self.assertEqual(out.shape, (2, 3))

        idxa = tk.zeros((3, 1, 2), tk.int32)
        idxb = tk.zeros((2, 3, 1, 2), tk.int32)
        out = tk.vmap(gather, (0, None, 0))(a, idxa, idxb)
        self.assertEqual(out.shape, (2, 3, 1, 2))

        idxa = tk.zeros((3, 1, 2), tk.int32)
        idxb = tk.zeros((3, 1, 2, 2), tk.int32)
        out = tk.vmap(gather, (0, None, 3))(a, idxa, idxb)
        self.assertEqual(out.shape, (2, 3, 1, 2))

    def test_vmap_scatter(self):
        def scatter(a):
            a[tk.array(0)] = tk.array(0.0)
            return a

        a = tk.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        out = tk.vmap(scatter)(a)
        expected = tk.array([[0.0, 2.0, 3.0], [0.0, 3.0, 4.0]])
        self.assertTrue(tk.allclose(out, expected))

        out = tk.vmap(scatter, in_axes=(1,), out_axes=1)(a)
        expected = tk.array([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
        self.assertTrue(tk.allclose(out, expected))

        def scatter_add(a):
            return a.at[tk.array(0)].add(tk.array(1.0))

        a = tk.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        out = tk.vmap(scatter_add)(a)
        expected = tk.array([[2.0, 2.0, 3.0], [3.0, 3.0, 4.0]])
        self.assertTrue(tk.allclose(out, expected))

        out = tk.vmap(scatter_add, in_axes=(1,), out_axes=1)(a)
        expected = tk.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        self.assertTrue(tk.allclose(out, expected))

        # Multiple indices
        def scatter(a):
            a[tk.array([0, 1]), tk.array([0, 1])] = tk.array((1.0, 1.0))
            return a

        a = tk.zeros((3, 3, 3))

        expected = tk.repeat(scatter(tk.zeros((3, 3)))[None], 3, axis=0)
        out = tk.vmap(scatter, in_axes=(0,), out_axes=0)(a)
        self.assertTrue(tk.allclose(out, expected))

        expected = tk.zeros((3, 3, 3))
        expected[0, :, 0] = 1
        expected[1, :, 1] = 1
        out = tk.vmap(scatter, in_axes=(1,), out_axes=1)(a)
        self.assertTrue(tk.allclose(out, expected))

        expected = tk.zeros((3, 3, 3))
        expected[0, 0, :] = 1
        expected[1, 1, :] = 1
        out = tk.vmap(scatter, in_axes=(2,), out_axes=2)(a)
        self.assertTrue(tk.allclose(out, expected))

        # vmap over src and indices
        def scatter(a, idx):
            a[idx] = tk.array(1.0)
            return a

        a = tk.zeros((3, 4))
        idx = tk.array([0, 1, 2])
        out = tk.vmap(scatter, in_axes=(0, 0), out_axes=0)(a, idx)
        self.assertTrue(tk.allclose(out, tk.eye(n=3, m=4)))

        # vmap over only indices
        out = tk.vmap(scatter, in_axes=(None, 0), out_axes=0)(a, idx)
        expected = tk.zeros((3, 3, 4))
        expected[0, 0] = 1
        expected[1, 1] = 1
        expected[2, 2] = 1
        self.assertTrue(tk.allclose(out, expected))

        # vmap over src, indices, updates
        def scatter(a, idx, updates):
            a[idx] = updates
            return a

        a = tk.zeros((3, 4))
        idx = tk.array([0, 1, 2])
        updates = tk.array([1, 2, 3])
        out = tk.vmap(scatter, in_axes=(0, 0, 0), out_axes=0)(a, idx, updates)
        expected = tk.diag(tk.array([1, 2, 3]), k=-1)[1:]
        self.assertTrue(tk.allclose(out, expected))

        # vmap over only updates
        def scatter(a, idx, updates):
            a[idx] = updates
            return a

        a = tk.zeros((3, 4))
        idx = tk.array([0])
        updates = tk.array([1, 2, 3])
        out = tk.vmap(scatter, in_axes=(None, None, 0), out_axes=0)(a, idx, updates)
        expected = tk.zeros((3, 3, 4))
        expected[:, 0] = tk.array([1, 2, 3])[:, None]
        self.assertTrue(tk.allclose(out, expected))

    def test_vmap_const_func(self):
        a = tk.random.uniform(shape=(2, 3, 4))
        b = tk.random.uniform(shape=(4, 3))

        def const_func(a, b):
            return tk.array(2)

        out = tk.vmap(const_func, in_axes=(0, None))(a, b)
        self.assertTrue(tk.array_equal(tk.full((2,), 2), out))
        out = tk.vmap(const_func, in_axes=(None, 0))(a, b)
        self.assertTrue(tk.array_equal(tk.full((4,), 2), out))
        out = tk.vmap(const_func, in_axes=(1, 1))(a, b)
        self.assertTrue(tk.array_equal(tk.full((3,), 2), out))

        with self.assertRaises(ValueError):
            out = tk.vmap(const_func, in_axes=(None, None))(a, b)

        with self.assertRaises(ValueError):
            out = tk.vmap(const_func, in_axes=(0, 0))(a, b)

    def test_vmap_concatenate(self):
        x = tk.random.uniform(shape=(2, 2, 2))

        def cat_fun(x, y):
            return tk.concatenate([x, y], axis=1)

        def cat_constant(x):
            y = tk.ones((2, 1))
            return tk.concatenate([x, y], 1)

        out = tk.vmap(cat_fun, in_axes=(0, 2))(x, x)
        target = tk.stack(
            [tk.concatenate([x[i], x[:, :, i]], axis=1) for i in range(2)]
        )
        self.assertTrue(tk.array_equal(out, target))

        out = tk.vmap(cat_constant)(x)
        target = tk.concatenate([x, tk.ones((2, 2, 1))], axis=2)
        self.assertTrue(tk.array_equal(out, target))

    def test_vmap_take_along_axis(self):
        a = tk.zeros((4, 5, 1))
        idx = tk.zeros((2, 4, 1), tk.int32)

        def fun(a, idx):
            return tk.take_along_axis(a, idx, axis=0)

        out = tk.vmap(fun, in_axes=(0, 1))(a, idx)
        self.assertEqual(out.shape, (4, 2, 1))

        idx = tk.zeros((2, 1), tk.int32)

        out = tk.vmap(fun, in_axes=(0, None))(a, idx)
        self.assertEqual(out.shape, (4, 2, 1))

        a = tk.zeros((5, 1))
        idx = tk.zeros((4, 2, 1), tk.int32)

        out = tk.vmap(fun, in_axes=(None, 0))(a, idx)
        self.assertEqual(out.shape, (4, 2, 1))

        a = tk.zeros((4, 5, 3))
        idx = tk.zeros((2, 2, 1, 3), tk.int32)

        out = tk.vmap(fun, in_axes=(None, 0))(a, idx)
        self.assertEqual(out.shape, (2, 2, 5, 3))

    def test_vmap_put_along_axis(self):
        a = tk.zeros((4, 5, 1))
        idx = tk.ones((2, 4, 1), tk.int32)
        upd = tk.ones((2, 4, 1))

        def fun(a, idx, upd):
            return tk.put_along_axis(a, idx, upd, axis=0)

        out = tk.vmap(fun, in_axes=(0, 1, 1))(a, idx, upd)
        self.assertEqual(out.shape, (4, 5, 1))

        upd = tk.ones((2, 1))
        out = tk.vmap(fun, in_axes=(0, 1, None))(a, idx, upd)
        self.assertEqual(out.shape, (4, 5, 1))

        idx = tk.ones((2, 1), tk.int32)
        upd = tk.ones((2, 1))
        out = tk.vmap(fun, in_axes=(0, None, None))(a, idx, upd)
        self.assertEqual(out.shape, (4, 5, 1))

        a = tk.zeros((5, 1))
        idx = tk.ones((2, 4, 1), tk.int32)
        upd = tk.ones((2, 4, 1))
        out = tk.vmap(fun, in_axes=(None, 1, 1))(a, idx, upd)
        self.assertEqual(out.shape, (4, 5, 1))

    def test_vmap_split_vmap(self):
        def fun(x):
            a, b = tk.split(x, 2, 1)
            return tk.concatenate([b, a], 1)

        x = tk.ones((5, 6, 7))
        y = tk.ones((5, 4, 6, 7))
        fx = fun(x)
        fy = tk.vmap(fun, in_axes=1)(y)
        self.assertEqual(fx.shape, (5, 6, 7))
        self.assertEqual(fy.shape, (4, 5, 6, 7))

    def test_leaks(self):
        gc.collect()
        tk.synchronize()
        if tk.metal.is_available():
            mem_pre = tk.get_active_memory()
        else:
            mem_pre = 0

        def outer():
            d = {}

            def f(x):
                return d["x"]

            d["f"] = tk.vmap(f)
            d["x"] = tk.array([0] * 1000)

        for _ in range(5):
            outer()
            gc.collect()

        tk.synchronize()
        if tk.metal.is_available():
            mem_post = tk.get_active_memory()
        else:
            mem_post = 0

        self.assertEqual(mem_pre, mem_post)

    def test_vmap_flatten(self):
        def fun(x):
            return tk.flatten(x, 0, 1)

        x = tk.zeros((2, 3, 4))

        self.assertEqual(tk.vmap(fun)(x).shape, (2, 12))
        self.assertEqual(tk.vmap(fun, in_axes=(1,))(x).shape, (3, 8))
        self.assertEqual(tk.vmap(fun, in_axes=(2,))(x).shape, (4, 6))

    def test_vmap_conv(self):
        # vmap input only
        x = tk.random.uniform(shape=(2, 2, 5, 4))
        w = tk.random.uniform(shape=(8, 3, 4))

        expected = tk.stack([tk.conv1d(xi, w) for xi in x])
        out = tk.vmap(tk.conv1d, in_axes=(0, None))(x, w)
        self.assertTrue(tk.allclose(expected, out))

        x = tk.moveaxis(x, 0, 2)
        out = tk.vmap(tk.conv1d, in_axes=(2, None))(x, w)
        self.assertTrue(tk.allclose(expected, out))

        # vmap weights only
        x = tk.random.uniform(shape=(2, 5, 4))
        w = tk.random.uniform(shape=(3, 8, 3, 4))

        expected = tk.stack([tk.conv1d(x, wi) for wi in w])
        out = tk.vmap(tk.conv1d, in_axes=(None, 0))(x, w)
        self.assertTrue(tk.allclose(expected, out))

        w = tk.moveaxis(w, 0, 1)
        out = tk.vmap(tk.conv1d, in_axes=(None, 1))(x, w)
        self.assertTrue(tk.allclose(expected, out))

        # vmap weights and input
        x = tk.random.uniform(shape=(3, 2, 5, 4))
        w = tk.random.uniform(shape=(3, 8, 3, 4))

        expected = tk.stack([tk.conv1d(xi, wi) for xi, wi in zip(x, w)])
        out = tk.vmap(tk.conv1d, in_axes=(0, 0))(x, w)
        self.assertTrue(tk.allclose(expected, out))

        x = tk.random.uniform(shape=(2, 3, 5, 4))
        w = tk.random.uniform(shape=(8, 3, 4, 3))

        expected = tk.stack([tk.conv1d(x[:, i], w[..., i]) for i in range(3)])
        out = tk.vmap(tk.conv1d, in_axes=(1, 3))(x, w)
        self.assertTrue(tk.allclose(expected, out))

        # Test with groups
        x = tk.random.uniform(shape=(3, 2, 5, 8))
        w = tk.random.uniform(shape=(3, 2, 3, 4))

        def gconv(x, w):
            return tk.conv1d(x, w, groups=2)

        expected = tk.stack([gconv(xi, wi) for xi, wi in zip(x, w)])
        out = tk.vmap(gconv, in_axes=(0, 0))(x, w)
        self.assertTrue(tk.allclose(expected, out))

    def test_vmap_pad(self):
        def pad2d(x, value=0.0):
            return tk.pad(x, ((1, 2), (0, 1)), constant_values=value)

        x = tk.arange(24, dtype=tk.float32).reshape(2, 3, 4)

        expected = tk.stack([pad2d(xi) for xi in x])
        out = tk.vmap(pad2d, in_axes=0)(x)
        self.assertTrue(tk.array_equal(out, expected))

        expected = tk.stack([pad2d(x[:, i, :]) for i in range(x.shape[1])])
        out = tk.vmap(pad2d, in_axes=1)(x)
        self.assertTrue(tk.array_equal(out, expected))

        expected = tk.stack([pad2d(x[:, :, i]) for i in range(x.shape[2])], axis=2)
        out = tk.vmap(pad2d, in_axes=-1, out_axes=-1)(x)
        self.assertTrue(tk.array_equal(out, expected))

        nested = tk.vmap(tk.vmap(lambda y: tk.pad(y, (1, 1))))
        out = nested(x)
        expected = tk.pad(x, ((0, 0), (0, 0), (1, 1)))
        self.assertTrue(tk.array_equal(out, expected))

        out = tk.vmap(
            lambda a, v: tk.pad(a, ((1, 1), (1, 1)), constant_values=v),
            in_axes=(0, None),
        )(x, tk.array(5.0))
        expected = tk.stack(
            [tk.pad(xi, ((1, 1), (1, 1)), constant_values=tk.array(5.0)) for xi in x]
        )
        self.assertTrue(tk.array_equal(out, expected))

        pad_values = tk.array([3.0, 4.0])
        with self.assertRaises(ValueError):
            tk.vmap(lambda a, v: tk.pad(a, ((1, 1), (1, 1)), constant_values=v))(
                x, pad_values
            )

    def test_vmap_types(self):

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

        vmap_transform = tk.vmap(transform)
        vmap_transform_tuple = tk.vmap(transform_tuple)
        vmap_transform_vector = tk.vmap(transform_vector)

        x_batch_tuple = (tk.array([1, 2, 3]), tk.array([4, 5, 6]))
        out1 = vmap_transform_tuple(x_batch_tuple)

        self.assertTrue(isinstance(out1, tuple))
        self.assertTrue(tk.array_equal(out1[0], tk.array([11, 12, 13])))
        self.assertTrue(tk.array_equal(out1[1], tk.array([40, 50, 60])))

        x_batch = State(tk.array([1, 2, 3]), tk.array([4, 5, 6]))
        out2 = vmap_transform(x_batch)
        self.assertTrue(isinstance(out2, State))
        self.assertTrue(tk.array_equal(out2.a, tk.array([11, 12, 13])))
        self.assertTrue(tk.array_equal(out2.b, tk.array([40, 50, 60])))

        x_batch_vector = Vector([tk.array([1, 2, 3]), tk.array([4, 5, 6])])
        out3 = vmap_transform_vector(x_batch_vector)
        self.assertTrue(isinstance(out3, Vector))
        self.assertTrue(tk.array_equal(out3[0], tk.array([11, 12, 13])))
        self.assertTrue(tk.array_equal(out3[1], tk.array([40, 50, 60])))

    def test_vmap_masked_scatter(self):
        def scatter_fn(x, m, src):
            x[m] = src
            return x

        # Batched sources
        a = tk.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        mask = tk.array([[False, True, True, True], [True, False, True, True]])
        src = tk.array([[1, 2, 3], [4, 5, 6]])

        expected = tk.array([[10, 1, 2, 3], [4, 60, 5, 6]])
        vmap_scatter = tk.vmap(scatter_fn, in_axes=(0, 0, 0))
        out = vmap_scatter(a, mask, src)
        self.assertTrue(tk.array_equal(expected, out))

        # Shared source across batch (matching mask populations)
        a = tk.array([[0, 0, 0], [5, 5, 5]])
        mask = tk.array([[True, False, True], [False, True, True]])
        src = tk.array([9, 8])

        expected = tk.array([[9, 0, 8], [5, 9, 8]])
        vmap_scatter = tk.vmap(scatter_fn, in_axes=(0, 0, None))
        out = vmap_scatter(a, mask, src)
        self.assertTrue(tk.array_equal(expected, out))

        # Shared destination with batched mask and sources
        a = tk.array([10, 20, 30, 40])
        mask = tk.array([[True, False, False, True], [False, True, True, False]])
        src = tk.array([[1, 2], [3, 4]])

        expected = tk.array([[1, 20, 30, 2], [10, 3, 4, 40]])
        vmap_scatter = tk.vmap(scatter_fn, in_axes=(None, 0, 0))
        out = vmap_scatter(a, mask, src)
        self.assertTrue(tk.array_equal(expected, out))

        # Shared mask across batch with batched sources
        a = tk.array([[0, 0, 0, 0], [10, 20, 30, 40]])
        mask = tk.array([True, False, True, False])
        src = tk.array([[7, 8], [9, 10]])

        expected = tk.array([[7, 0, 8, 0], [9, 20, 10, 40]])
        vmap_scatter = tk.vmap(scatter_fn, in_axes=(0, None, 0))
        out = vmap_scatter(a, mask, src)
        self.assertTrue(tk.array_equal(expected, out))

        # Uneven mask populations with scalar broadcast
        a = tk.array([[0.0, 0.0, 0.0, 0.0], [10.0, 20.0, 30.0, 40.0]])
        mask = tk.array([[True, False, True, True], [False, True, False, False]])
        shared_src = tk.array(1.5)

        expected = tk.array(
            [[1.5, 0.0, 1.5, 1.5], [10.0, 1.5, 30.0, 40.0]], dtype=a.dtype
        )
        vmap_scatter = tk.vmap(scatter_fn, in_axes=(0, 0, None))
        out = vmap_scatter(a, mask, shared_src)
        self.assertTrue(tk.array_equal(expected, out))

        # Shared src with identical masks must restart for each batch
        a = tk.array([[0, 0, 0, 0, 0], [10, 20, 30, 40, 50]])
        mask = tk.array(
            [[True, True, True, False, False], [True, True, True, False, False]]
        )
        src = tk.array([1, 2, 3, 4, 5])

        expected = tk.array([[1, 2, 3, 0, 0], [1, 2, 3, 40, 50]])
        vmap_scatter = tk.vmap(scatter_fn, in_axes=(0, 0, None))
        out = vmap_scatter(a, mask, src)
        self.assertTrue(tk.array_equal(expected, out))

        # Double vmap
        a = tk.zeros((8, 8, 8))
        mask = tk.random.normal((8, 8, 8)) > 0
        src = tk.random.normal((8, 8))
        expected = tk.stack(
            [
                tk.stack(
                    [scatter_fn(a[i, j] + 0, mask[i, j], src[i]) for j in range(8)]
                )
                for i in range(8)
            ]
        )
        double_scatter = tk.vmap(
            tk.vmap(scatter_fn, in_axes=(0, 0, None)), in_axes=(0, 0, 0)
        )
        out = double_scatter(a + 0, mask, src)
        self.assertTrue(tk.array_equal(expected, out))

    def test_broadcast_axes_vmap(self):
        # Broadcast axes requires shapeless compile to properly test

        counter = [0]

        def fn(x, y):
            counter[0] += 1
            return tk.matmul(x, y)

        x = tk.random.normal((2, 3, 1, 4, 5))
        y = tk.random.normal((1, 2, 5, 6))
        z = tk.random.normal((3, 2, 1, 4, 5))
        w = tk.random.normal((2, 3, 5, 6))

        vmap_fn = tk.vmap(fn, in_axes=(0, 1))
        cvmap_fn = tk.compile(vmap_fn, shapeless=True)

        expected = vmap_fn(x, y)
        out = cvmap_fn(x, y)
        self.assertTrue(tk.array_equal(expected, out))
        self.assertEqual(2, counter[0])

        expected = vmap_fn(z, w)
        out = cvmap_fn(z, w)
        self.assertTrue(tk.array_equal(expected, out))
        self.assertEqual(3, counter[0])

        x = tk.random.normal((2, 3, 1, 4, 5))
        y = tk.random.normal((1, 2, 5, 6))
        z = tk.random.normal((2, 3, 1, 7, 2))
        w = tk.random.normal((1, 2, 2, 3))

        vmap_fn = tk.vmap(fn, in_axes=(0, None))
        cvmap_fn = tk.compile(vmap_fn, shapeless=True)

        expected = vmap_fn(x, y)
        out = cvmap_fn(x, y)
        self.assertTrue(tk.array_equal(expected, out))
        self.assertEqual(5, counter[0])

        expected = vmap_fn(z, w)
        out = cvmap_fn(z, w)
        self.assertTrue(tk.array_equal(expected, out))
        self.assertEqual(6, counter[0])

    def test_vmap_sort(self):
        a = tk.random.uniform(shape=(3, 5))
        expected = tk.stack([tk.sort(a[:, i]) for i in range(a.shape[1])], axis=1)
        for axis in (0, -1):
            out = tk.vmap(lambda x: tk.sort(x, axis=axis), in_axes=1, out_axes=1)(a)
            self.assertTrue(tk.array_equal(out, expected))

    def test_vmap_argsort(self):
        a = tk.random.uniform(shape=(3, 5))
        expected = tk.stack([tk.argsort(a[:, i]) for i in range(a.shape[1])], axis=1)
        for axis in (0, -1):
            out = tk.vmap(lambda x: tk.argsort(x, axis=axis), in_axes=1, out_axes=1)(a)
            self.assertTrue(tk.array_equal(out, expected))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
