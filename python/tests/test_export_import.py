# Copyright © 2024 Apple Inc.

import gc
import json
import os
import tempfile
import unittest

import tiki as tk
import tiki.nn as nn
import tiki_tests


class TestExportImport(tiki_tests.TIKITestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_basic_export_import(self):
        path = os.path.join(self.test_dir, "fn.tkfn")

        # Function with no inputs
        def fun():
            return tk.zeros((3, 3))

        tk.export_function(path, fun)
        imported = tk.import_function(path)

        expected = fun()
        (out,) = imported()
        self.assertTrue(tk.array_equal(out, expected))

        # Simple function with inputs
        def fun(x):
            return tk.abs(tk.sin(x))

        inputs = tk.array([1.0, 2.0, 3.0, 4.0, 5.0])

        tk.export_function(path, fun, inputs)
        imported = tk.import_function(path)

        expected = fun(inputs)
        (out,) = imported(inputs)
        self.assertTrue(tk.allclose(out, expected))

        # Inputs in a list or tuple
        def fun(x):
            x = tk.abs(tk.sin(x))
            return x

        tk.export_function(path, fun, [inputs])
        imported = tk.import_function(path)

        expected = fun(inputs)
        (out,) = imported([inputs])
        self.assertTrue(tk.allclose(out, expected))

        (out,) = imported(inputs)
        self.assertTrue(tk.allclose(out, expected))

        tk.export_function(path, fun, (inputs,))
        imported = tk.import_function(path)
        (out,) = imported((inputs,))
        self.assertTrue(tk.allclose(out, expected))

        # Outputs in a list
        def fun(x):
            return [tk.abs(tk.sin(x))]

        tk.export_function(path, fun, inputs)
        imported = tk.import_function(path)
        (out,) = imported(inputs)
        self.assertTrue(tk.allclose(out, expected))

        # Outputs in a tuple
        def fun(x):
            return (tk.abs(tk.sin(x)),)

        tk.export_function(path, fun, inputs)
        imported = tk.import_function(path)
        (out,) = imported(inputs)
        self.assertTrue(tk.allclose(out, expected))

        # Check throws on invalid inputs / outputs
        def fun(x):
            return tk.abs(x)

        with self.assertRaises(ValueError):
            tk.export_function(path, fun, "hi")

        with self.assertRaises(ValueError):
            tk.export_function(path, fun, tk.array(1.0), "hi")

        def fun(x):
            return tk.abs(x[0][0])

        with self.assertRaises(ValueError):
            tk.export_function(path, fun, [[tk.array(1.0)]])

        def fun():
            return (tk.zeros((3, 3)), 1)

        with self.assertRaises(ValueError):
            tk.export_function(path, fun)

        def fun():
            return (tk.zeros((3, 3)), [tk.zeros((3, 3))])

        with self.assertRaises(ValueError):
            tk.export_function(path, fun)

        def fun(x, y):
            return x + y

        tk.export_function(path, fun, tk.array(1.0), tk.array(1.0))
        imported = tk.import_function(path)

        with self.assertRaises(ValueError):
            imported(tk.array(1.0), 1.0)

        with self.assertRaises(ValueError):
            imported(tk.array(1.0), tk.array(1.0), tk.array(1.0))

        with self.assertRaises(ValueError):
            imported(tk.array(1.0), [tk.array(1.0)])

    def test_export_random_sample(self):
        path = os.path.join(self.test_dir, "fn.tkfn")

        tk.random.seed(5)

        def fun():
            return tk.random.uniform(shape=(3,))

        tk.export_function(path, fun)
        imported = tk.import_function(path)

        (out,) = imported()

        tk.random.seed(5)
        expected = fun()

        self.assertTrue(tk.array_equal(out, expected))

    def test_export_with_kwargs(self):
        path = os.path.join(self.test_dir, "fn.tkfn")

        def fun(x, z=None):
            out = x
            if z is not None:
                out += z
            return out

        x = tk.array([1, 2, 3])
        y = tk.array([1, 1, 0])
        z = tk.array([2, 2, 2])

        tk.export_function(path, fun, (x,), {"z": z})
        imported_fun = tk.import_function(path)

        with self.assertRaises(ValueError):
            imported_fun(x, z)

        with self.assertRaises(ValueError):
            imported_fun(x, y=z)

        with self.assertRaises(ValueError):
            imported_fun((x,), {"y": z})

        out = imported_fun(x, z=z)[0]
        self.assertTrue(tk.array_equal(out, tk.array([3, 4, 5])))

        out = imported_fun((x,), {"z": z})[0]
        self.assertTrue(tk.array_equal(out, tk.array([3, 4, 5])))

        tk.export_function(path, fun, x, z=z)
        imported_fun = tk.import_function(path)
        out = imported_fun(x, z=z)[0]
        self.assertTrue(tk.array_equal(out, tk.array([3, 4, 5])))

        out = imported_fun((x,), {"z": z})[0]
        self.assertTrue(tk.array_equal(out, tk.array([3, 4, 5])))

        # Only specify kwargs
        tk.export_function(path, fun, x=x, z=z)
        imported_fun = tk.import_function(path)
        with self.assertRaises(ValueError):
            out = imported_fun(x, z=z)[0]

        out = imported_fun(x=x, z=z)[0]
        self.assertTrue(tk.array_equal(out, tk.array([3, 4, 5])))

        out = imported_fun({"x": x, "z": z})[0]
        self.assertTrue(tk.array_equal(out, tk.array([3, 4, 5])))

    def test_export_variable_inputs(self):
        path = os.path.join(self.test_dir, "fn.tkfn")

        def fun(x, y, z=None):
            out = x + y
            if z is not None:
                out += z
            return out

        with tk.exporter(path, fun) as exporter:
            exporter(tk.array([1, 2, 3]), tk.array([1, 1, 1]))
            exporter(tk.array([1, 2, 3]), tk.array([1, 1, 1]), z=tk.array([2]))

        with self.assertRaises(RuntimeError):
            exporter(tk.array([1, 2, 3, 4]), tk.array([1, 1, 1, 1]))

        imported_fun = tk.import_function(path)
        out = imported_fun(tk.array([1, 2, 3]), tk.array([1, 1, 1]))[0]
        self.assertTrue(tk.array_equal(out, tk.array([2, 3, 4])))

        out = imported_fun(tk.array([1, 2, 3]), tk.array([1, 1, 1]), z=tk.array([2]))[0]
        self.assertTrue(tk.array_equal(out, tk.array([4, 5, 6])))

        with self.assertRaises(ValueError):
            imported_fun(tk.array([1, 2, 3, 4]), tk.array([1, 1, 1, 1]))

        # A function with a large constant
        constant = tk.zeros((16, 2048))
        tk.eval(constant)

        def fun(*args):
            return constant + sum(args)

        with tk.exporter(path, fun) as exporter:
            for i in range(5):
                exporter(*[tk.array(1)] * i)

        # Check the exported file size < constant size + small amount
        constants_size = constant.nbytes + 8192
        self.assertTrue(os.path.getsize(path) < constants_size)

    def test_leaks(self):
        path = os.path.join(self.test_dir, "fn.tkfn")
        tk.synchronize()
        if tk.metal.is_available():
            mem_pre = tk.get_active_memory()
        else:
            mem_pre = 0

        def outer():
            d = {}

            def f(x):
                return d["x"]

            d["f"] = tk.exporter(path, f)
            d["x"] = tk.array([0] * 1000)

        for _ in range(5):
            outer()
            gc.collect()

        if tk.metal.is_available():
            mem_post = tk.get_active_memory()
        else:
            mem_post = 0

        self.assertEqual(mem_pre, mem_post)

    def test_export_import_shapeless(self):
        path = os.path.join(self.test_dir, "fn.tkfn")

        def fun(*args):
            return sum(args)

        with tk.exporter(path, fun, shapeless=True) as exporter:
            exporter(tk.array(1))
            exporter(tk.array(1), tk.array(2))
            exporter(tk.array(1), tk.array(2), tk.array(3))

        f2 = tk.import_function(path)
        self.assertEqual(f2(tk.array(1))[0].item(), 1)
        self.assertEqual(f2(tk.array(1), tk.array(1))[0].item(), 2)
        self.assertEqual(f2(tk.array(1), tk.array(1), tk.array(1))[0].item(), 3)
        with self.assertRaises(ValueError):
            f2(tk.array(10), tk.array([5, 10, 20]))

    def test_export_scatter_gather(self):
        path = os.path.join(self.test_dir, "fn.tkfn")

        def fun(a, b):
            return tk.take_along_axis(a, b, axis=0)

        x = tk.random.uniform(shape=(4, 4))
        y = tk.array([[0, 1, 2, 3], [1, 2, 0, 3]])
        tk.export_function(path, fun, (x, y))
        imported_fun = tk.import_function(path)
        expected = fun(x, y)
        out = imported_fun(x, y)[0]
        self.assertTrue(tk.array_equal(expected, out))

        def fun(a, b, c):
            return tk.put_along_axis(a, b, c, axis=0)

        x = tk.random.uniform(shape=(4, 4))
        y = tk.array([[0, 1, 2, 3], [1, 2, 0, 3]])
        z = tk.random.uniform(shape=(2, 4))
        tk.export_function(path, fun, (x, y, z))
        imported_fun = tk.import_function(path)
        expected = fun(x, y, z)
        out = imported_fun(x, y, z)[0]
        self.assertTrue(tk.array_equal(expected, out))

    def test_export_searchsorted(self):
        path = os.path.join(self.test_dir, "fn.tkfn")

        # both sides, since the side is the primitive's only state and a lost
        # state would still round trip for the default
        for side in ("left", "right"):

            def fun(a, v):
                return tk.searchsorted(a, v, side=side)

            x = tk.sort(tk.random.uniform(shape=(32,)))
            y = tk.random.uniform(shape=(3, 5))
            tk.export_function(path, fun, (x, y))
            imported_fun = tk.import_function(path)
            expected = fun(x, y)
            out = imported_fun(x, y)[0]
            self.assertTrue(tk.array_equal(expected, out))

    def test_export_conv(self):
        path = os.path.join(self.test_dir, "fn.tkfn")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(
                    3, 16, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.c2 = nn.Conv2d(
                    16, 16, kernel_size=3, stride=2, padding=1, bias=False
                )
                self.c3 = nn.Conv2d(
                    16, 16, kernel_size=3, stride=1, padding=2, bias=False
                )

            def __call__(self, x):
                return self.c3(self.c2(self.c1(x)))

        model = Model()
        tk.eval(model.parameters())

        def forward(x):
            return model(x)

        input_data = tk.random.normal(shape=(4, 32, 32, 3))
        tk.export_function(path, forward, input_data)

        imported_fn = tk.import_function(path)
        out = imported_fn(input_data)[0]
        expected = forward(input_data)
        self.assertTrue(tk.allclose(expected, out))

    def test_export_conv_shapeless(self):
        # Conv1d (NLC)
        path = os.path.join(self.test_dir, "conv1d.tkfn")

        class M1(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv1d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)

            def __call__(self, x):
                return self.c(x)

        m1 = M1()
        tk.eval(m1.parameters())

        def f1(x):
            return m1(x)

        x = tk.random.normal(shape=(4, 64, 3))
        tk.export_function(path, f1, x, shapeless=True)
        f1_imp = tk.import_function(path)
        for shape in [(4, 64, 3), (1, 33, 3), (2, 128, 3)]:
            xt = tk.random.normal(shape=shape)
            self.assertTrue(tk.allclose(f1_imp(xt)[0], f1(xt)))

        # Conv2d (NHWC)
        path = os.path.join(self.test_dir, "conv2d.tkfn")

        class M2(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1, bias=False)

            def __call__(self, x):
                return self.c(x)

        m2 = M2()
        tk.eval(m2.parameters())

        def f2(x):
            return m2(x)

        x = tk.random.normal(shape=(2, 32, 32, 3))
        tk.export_function(path, f2, x, shapeless=True)
        f2_imp = tk.import_function(path)
        for shape in [(2, 32, 32, 3), (1, 31, 31, 3), (4, 64, 48, 3)]:
            xt = tk.random.normal(shape=shape)
            self.assertTrue(tk.allclose(f2_imp(xt)[0], f2(xt)))

        # Conv3d (NDHWC)
        path = os.path.join(self.test_dir, "conv3d.tkfn")

        class M3(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv3d(2, 4, kernel_size=3, stride=2, padding=1, bias=False)

            def __call__(self, x):
                return self.c(x)

        m3 = M3()
        tk.eval(m3.parameters())

        def f3(x):
            return m3(x)

        x = tk.random.normal(shape=(1, 8, 8, 8, 2))
        tk.export_function(path, f3, x, shapeless=True)
        f3_imp = tk.import_function(path)
        for shape in [(1, 8, 8, 8, 2), (2, 7, 8, 9, 2), (1, 16, 16, 4, 2)]:
            xt = tk.random.normal(shape=shape)
            self.assertTrue(tk.allclose(f3_imp(xt)[0], f3(xt)))

        # Grouped Conv2d (NHWC)
        path = os.path.join(self.test_dir, "conv2d_grouped.tkfn")

        class MG(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(
                    4, 6, kernel_size=3, stride=2, padding=1, groups=2, bias=False
                )

            def __call__(self, x):
                return self.c(x)

        mg = MG()
        tk.eval(mg.parameters())

        def fg(x):
            return mg(x)

        x = tk.random.normal(shape=(2, 32, 32, 4))
        tk.export_function(path, fg, x, shapeless=True)
        fg_imp = tk.import_function(path)
        for shape in [(2, 32, 32, 4), (1, 32, 32, 4), (3, 15, 20, 4)]:
            xt = tk.random.normal(shape=shape)
            self.assertTrue(tk.allclose(fg_imp(xt)[0], fg(xt)))

    def test_export_control_flow(self):

        def fun(x, y):
            if y.shape[0] <= 2:
                return x + y
            else:
                return x + 2 * y

        for y in (tk.array([1, 2, 3]), tk.array([1, 2])):
            for shapeless in (True, False):
                with self.subTest(y=y, shapeless=shapeless):
                    x = tk.array(1)
                    export_path = os.path.join(self.test_dir, "control_flow.tkfn")
                    tk.export_function(export_path, fun, x, y, shapeless=shapeless)

                    imported_fn = tk.import_function(export_path)
                    self.assertTrue(tk.array_equal(imported_fn(x, y)[0], fun(x, y)))

    def test_export_quantized_model(self):
        for shapeless in (True, False):
            with self.subTest(shapeless=shapeless):
                model = nn.Sequential(
                    nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1024)
                )
                model.eval()
                tk.eval(model.parameters())
                input_data = tk.ones(shape=(512, 1024))
                nn.quantize(model)
                self.assertTrue(isinstance(model.layers[0], nn.QuantizedLinear))
                self.assertTrue(isinstance(model.layers[2], nn.QuantizedLinear))
                tk.eval(model.parameters())

                export_path = os.path.join(self.test_dir, "quantized_linear.tkfn")
                tk.export_function(export_path, model, input_data, shapeless=shapeless)

                imported_fn = tk.import_function(export_path)
                self.assertTrue(
                    tk.array_equal(imported_fn(input_data)[0], model(input_data))
                )

    def test_export_kwarg_ordering(self):
        path = os.path.join(self.test_dir, "fun.tkfn")

        def fn(x, y):
            return x - y

        tk.export_function(path, fn, x=tk.array(1.0), y=tk.array(1.0))
        imported = tk.import_function(path)
        out = imported(x=tk.array(2.0), y=tk.array(3.0))[0]
        self.assertEqual(out.item(), -1.0)
        out = imported(y=tk.array(2.0), x=tk.array(3.0))[0]
        self.assertEqual(out.item(), 1.0)

    def test_export_with_callback(self):

        def fn(x, y):
            return tk.log(tk.abs(x - y)).astype(tk.int32)

        n_in = None
        n_out = None
        n_const = None
        keywords = None
        primitives = []
        primitive_args = []

        def callback(args):
            nonlocal n_in, n_out, n_const, keywords, primitives
            t = args["type"]
            if t == "inputs":
                n_in = len(args["inputs"])
            elif args["type"] == "outputs":
                n_out = len(args["outputs"])
            elif args["type"] == "keyword_inputs":
                keywords = args["keywords"]
            elif t == "constants":
                n_const = len(args["constants"])
            elif t == "primitive":
                primitives.append(args["name"])
                primitive_args.append(args["arguments"])

        tk.export_function(callback, fn, tk.array(1.0), y=tk.array(1.0))
        self.assertEqual(n_in, 2)
        self.assertEqual(n_out, 1)
        self.assertEqual(n_const, 0)
        self.assertEqual(len(keywords), 1)
        self.assertEqual(keywords[0][0], "y")
        self.assertEqual(primitives, ["Subtract", "Abs", "Log", "AsType"])
        self.assertEqual(primitive_args[0], [])
        self.assertEqual(primitive_args[1], [])
        self.assertEqual(primitive_args[2], [2])
        self.assertEqual(primitive_args[3], [tk.int32])

    @unittest.skipIf(not tk.is_available(tk.gpu), "No GPU available")
    def test_export_import_custom_kernel(self):
        if tk.metal.is_available():
            source = """
                uint elem = thread_position_in_grid.x;
                out1[elem] = a[elem];
            """
            custom_kernel = tk.fast.metal_kernel
        elif tk.cuda.is_available():
            source = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                out1[elem] = a[elem];
            """
            custom_kernel = tk.fast.cuda_kernel

        kernel = custom_kernel(
            name="basic",
            input_names=["a"],
            output_names=["out1"],
            source=source,
        )

        def call(a):
            return kernel(
                inputs=[a],
                grid=(4, 1, 1),
                threadgroup=(2, 1, 1),
                output_shapes=[(2, 2)],
                output_dtypes=[tk.float32],
                stream=tk.gpu,
            )[0]

        tk.random.seed(7)
        a = tk.random.normal(shape=(2, 2))

        path = os.path.join(self.test_dir, "fn.tkfn")
        expected = call(a)
        tk.export_function(path, call, a)

        imported = tk.import_function(path)

        out = imported(a)[0]
        self.assertTrue(tk.allclose(expected, out))

    def test_export_custom_metal_kernel_without_evaluation(self):
        source = """
            uint elem = thread_position_in_grid.x;
            out[elem] = a[elem];
        """
        kernel = tk.fast.metal_kernel(
            name="export_only",
            input_names=["a"],
            output_names=["out"],
            source=source,
        )

        def call(a):
            return kernel(
                inputs=[a],
                grid=(a.size, 1, 1),
                threadgroup=(min(a.size, 256), 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=tk.gpu,
            )[0]

        a = tk.zeros((2, 2))
        for shapeless in (False, True):
            path = os.path.join(
                self.test_dir,
                f"metal_kernel_export_only_{shapeless}.tkfn",
            )
            tk.export_function(path, call, a, shapeless=shapeless)
            self.assertTrue(os.path.exists(path))

            # A shapeless import can't be called since CustomKernel does
            # not support shape inference
            if tk.metal.is_available() and not shapeless:
                imported = tk.import_function(path)
                self.assertTrue(tk.array_equal(imported(a)[0], call(a)))

        def call_cpu(a):
            return kernel(
                inputs=[a],
                grid=(a.size, 1, 1),
                threadgroup=(min(a.size, 256), 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=tk.cpu,
            )[0]

        path = os.path.join(self.test_dir, "metal_kernel_export_cpu.tkfn")
        with self.assertRaisesRegex(ValueError, "Only supports the GPU"):
            tk.export_function(path, call_cpu, a)

        if not tk.metal.is_available():
            with self.assertRaisesRegex(RuntimeError, "No Metal back-end"):
                call(a)
            with self.assertRaisesRegex(RuntimeError, "No Metal back-end"):
                tk.eval(tk.compile(call)(a))

    def test_export_custom_metal_kernel_with_math_mode(self):
        source = """
            uint elem = thread_position_in_grid.x;
            out[elem] = metal::exp(a[elem]);
        """
        kernel = tk.fast.metal_kernel(
            name="math_mode_export",
            input_names=["a"],
            output_names=["out"],
            source=source,
            compile_options={"math_mode": "safe"},
        )

        def call(a):
            return kernel(
                inputs=[a],
                grid=(a.size, 1, 1),
                threadgroup=(min(a.size, 256), 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=tk.gpu,
            )[0]

        a = tk.array([-float("inf"), 0.0])
        path = os.path.join(self.test_dir, "metal_kernel_math_mode.tkfn")
        tk.export_function(path, call, a)
        self.assertTrue(os.path.exists(path))

        if tk.metal.is_available():
            imported = tk.import_function(path)
            self.assertTrue(tk.array_equal(imported(a)[0], call(a)))

    def test_export_import_multi_with_constants(self):

        path = os.path.join(self.test_dir, "fn.tkfn")

        def fun(y):
            i = y.shape[0]
            x = tk.array(i)
            for j in range(10):
                x = x + tk.array(i + j)
            return x * y.sum()

        ys = [tk.array([1]), tk.array([1, 1]), tk.array([1, 1, 1])]

        with tk.exporter(path, fun) as exporter:
            for y in ys:
                exporter(y)

        imported = tk.import_function(path)
        for y in ys:
            self.assertEqual(imported(y)[0].item(), fun(y).item())

    def test_export_import_scatter_sum(self):
        def fun(x, y, z):
            return x.at[y].add(z)

        x = tk.array([1, 2, 3])
        y = tk.array([0, 0, 1])
        z = tk.array([1, 1, 1])
        path = os.path.join(self.test_dir, "fn.tkfn")
        tk.export_function(path, fun, x, y, z)

        imported = tk.import_function(path)
        self.assertTrue(tk.array_equal(imported(x, y, z)[0], fun(x, y, z)))

    def test_export_matmul_shapeless_mid_dim(self):
        path = os.path.join(self.test_dir, "matmul_shapeless.tkfn")

        E, H = 64, 17
        arr = tk.arange(E * H, dtype=tk.float32).reshape((E, H)) * (1.0 / (E * H))

        def fn(x):
            return tk.matmul(x, arr)

        sample = tk.zeros((1, 40, E), dtype=tk.float32)
        tk.export_function(path, fn, sample, shapeless=True)
        imported = tk.import_function(path)

        for seq_len in (40, 248, 623):
            with self.subTest(seq_len=seq_len):
                x = tk.arange(seq_len * E, dtype=tk.float32).reshape((1, seq_len, E))
                expected = fn(x)
                (y,) = imported(x)
                self.assertEqual(y.shape, (1, seq_len, H))
                self.assertTrue(tk.allclose(y, expected))

    def test_export_matmul_shapeless_batch_and_mid_dim(self):
        path = os.path.join(self.test_dir, "matmul_shapeless_batch.tkfn")

        B, E, H = 2, 32, 8
        arr = tk.arange(E * H, dtype=tk.float32).reshape((E, H)) * (1.0 / (E * H))

        def fn(x):
            return tk.matmul(x, arr)

        sample = tk.zeros((B, 10, E), dtype=tk.float32)
        tk.export_function(path, fn, sample, shapeless=True)
        imported = tk.import_function(path)

        for seq_len in (10, 50, 100):
            with self.subTest(seq_len=seq_len):
                x = tk.arange(B * seq_len * E, dtype=tk.float32).reshape(
                    (B, seq_len, E)
                )
                expected = fn(x)
                (y,) = imported(x)
                self.assertEqual(y.shape, (B, seq_len, H))
                self.assertTrue(tk.allclose(y, expected))

    def test_export_import_metadata(self):
        path = os.path.join(self.test_dir, "fn.tkfn")

        def fun(x):
            return tk.abs(x)

        x = tk.array([1.0, -2.0, 3.0])
        metadata = json.dumps({"name": "model", "params": 7_000_000_000, "lr": 0.1})

        tk.export_function(path, fun, x, metadata=metadata)

        imported = tk.import_function(path)
        self.assertTrue(tk.array_equal(imported(x)[0], fun(x)))

        imported, imported_metadata = tk.import_function(path, return_metadata=True)
        self.assertEqual(imported_metadata, metadata)
        self.assertEqual(json.loads(imported_metadata)["params"], 7_000_000_000)
        self.assertTrue(tk.array_equal(imported(x)[0], fun(x)))

        tk.export_function(path, fun, x)
        _, imported_metadata = tk.import_function(path, return_metadata=True)
        self.assertEqual(imported_metadata, "")

        # Metadata survives the per-trace header rewrite of a multi-trace export
        with tk.exporter(path, fun, metadata=metadata) as exporter:
            exporter(tk.array([1.0]))
            exporter(tk.array([1.0, 2.0]))
        _, imported_metadata = tk.import_function(path, return_metadata=True)
        self.assertEqual(imported_metadata, metadata)

        with self.assertRaises(TypeError):
            tk.export_function(path, fun, x, metadata={"name": "model"})

        with self.assertRaises(ValueError):
            tk.export_function(lambda x: None, fun, x, metadata=metadata)


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
