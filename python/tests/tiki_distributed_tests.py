# Copyright © 2025 Apple Inc.

import math

import tiki as tk
import tiki.nn as nn
import tiki_tests
from tiki.nn.layers.distributed import shard_inplace, shard_linear
from tiki.nn.utils import average_gradients, clip_grad_norm_sharded


class TIKIDistributedCommonTestCase(tiki_tests.TIKITestCase):
    def test_average_gradients(self):
        original_all_sum = tk.distributed.all_sum
        n_calls = 0
        xtype = None

        def new_all_sum(x, **kwargs):
            nonlocal n_calls
            nonlocal xtype

            n_calls += 1
            if xtype is not None:
                self.assertEqual(xtype, x.dtype)

            return original_all_sum(x, **kwargs)

        tk.distributed.all_sum = new_all_sum

        try:
            grads = [tk.ones(10) for i in range(10)]
            new_grads = average_gradients(grads)
            tk.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(tk.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 1)

            n_calls = 0
            new_grads = average_gradients(grads, all_reduce_size=4 * 50)
            tk.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(tk.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 2)

            n_calls = 0
            new_grads = average_gradients(grads, all_reduce_size=0)
            tk.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(tk.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 10)

        finally:
            tk.distributed.all_sum = original_all_sum

    def test_all_reduce(self):
        g = tk.distributed.init()
        dtypes = [
            (tk.int8, 0),
            (tk.uint8, 0),
            (tk.int32, 0),
            (tk.uint32, 0),
            (tk.float32, 1e-6),
            (tk.float16, 5e-3),
            (tk.bfloat16, 1e-1),
        ]
        sizes = [
            (7,),
            (10,),
            (1024,),
            (1024, 1024),
        ]
        key = tk.random.key(0)

        for dt, rtol in dtypes:
            for sh in sizes:
                x = (tk.random.uniform(shape=(g.size(),) + sh, key=key) * 10).astype(dt)

                # All sum
                y = tk.distributed.all_sum(x[g.rank()], group=g)
                z = x.sum(0)
                maxrelerror = (y - z).abs()
                if rtol > 0:
                    maxrelerror /= z.abs()
                maxrelerror = maxrelerror.max()
                self.assertLessEqual(maxrelerror, rtol)

                # All max
                y = tk.distributed.all_max(x[g.rank()], group=g)
                z = x.max(0)
                self.assertTrue(tk.all(y == z))

                # All min
                y = tk.distributed.all_min(x[g.rank()], group=g)
                z = x.min(0)
                self.assertTrue(tk.all(y == z))

    def test_donation(self):
        x = tk.random.normal((1024,))
        tk.eval(x)
        tk.synchronize()

        tk.reset_peak_memory()
        scale = tk.array(2.0)
        y = tk.distributed.all_sum(x)
        tk.eval(y)
        tk.synchronize()
        all_sum_only = tk.get_peak_memory()
        y = tk.distributed.all_sum(x) * scale
        tk.eval(y)
        tk.synchronize()
        all_sum_with_binary = tk.get_peak_memory()

        self.assertEqual(all_sum_only, all_sum_with_binary)

    def test_shard_linear(self):
        # Seed the prng to have the same inputs and weights generated everywhere
        tk.random.seed(0xF0F0F0F0)

        # Prepare inputs
        world = tk.distributed.init()
        part = (
            slice(None),
            slice(
                world.rank() * 1024 // world.size(),
                (world.rank() + 1) * 1024 // world.size(),
            ),
        )
        x = tk.random.normal((4, 1024))

        # Create and shard some linear layers
        lin = nn.Linear(1024, 1024, bias=True)
        slin1 = shard_linear(lin, "all-to-sharded")
        slin2 = shard_linear(lin, "sharded-to-all")
        y = lin(x)
        y1 = slin1(x)
        y2 = slin2(x[part])
        self.assertTrue(tk.allclose(y, y2, atol=self.atol, rtol=self.rtol))
        self.assertTrue(tk.allclose(y[part], y1, atol=self.atol, rtol=self.rtol))

        # And their quant versions (QuantizedMatmul is not supported on CUDA)
        if not tk.cuda.is_available():
            qlin = lin.to_quantized()
            slin1 = shard_linear(qlin, "all-to-sharded")
            slin2 = shard_linear(qlin, "sharded-to-all")
            y = qlin(x)
            y1 = slin1(x)
            y2 = slin2(x[part])
            self.assertTrue(tk.allclose(y, y2, atol=self.atol, rtol=self.rtol))
            self.assertTrue(tk.allclose(y[part], y1))

            # Test non-affine quantization modes (mxfp8)
            qlin_mxfp8 = lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
            self.assertEqual(qlin_mxfp8.mode, "mxfp8")

            slin1_mxfp8 = shard_linear(qlin_mxfp8, "all-to-sharded")
            slin2_mxfp8 = shard_linear(qlin_mxfp8, "sharded-to-all")

            # Verify mode is propagated
            self.assertEqual(slin1_mxfp8.mode, "mxfp8")
            self.assertEqual(slin2_mxfp8.mode, "mxfp8")

            # Verify biases parameter is not set for mxfp8
            self.assertIsNone(slin1_mxfp8.get("biases"))
            self.assertIsNone(slin2_mxfp8.get("biases"))

            y = qlin_mxfp8(x)
            y1 = slin1_mxfp8(x)
            y2 = slin2_mxfp8(x[part])
            self.assertTrue(tk.allclose(y, y2, atol=self.atol, rtol=self.rtol))
            self.assertTrue(tk.allclose(y[part], y1))

        # Check the backward works as expected
        def dummy_loss(model, x, y):
            return (model(x) * y).sum()

        mod = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
        )
        smod = nn.Sequential(
            shard_linear(mod.layers[0], "all-to-sharded"),
            shard_linear(mod.layers[1], "sharded-to-all"),
            shard_linear(mod.layers[2], "all-to-sharded"),
            shard_linear(mod.layers[3], "sharded-to-all"),
        )

        grad1 = nn.value_and_grad(mod, dummy_loss)
        grad2 = nn.value_and_grad(smod, dummy_loss)

        x = tk.random.normal((4, 128))
        y = tk.random.normal((4, 128))

        l1, g1 = grad1(mod, x, y)
        l2, g2 = grad2(smod, x, y)
        tk.eval(l1, g1, l2, g2)

        part = slice(
            world.rank() * 128 // world.size(), (world.rank() + 1) * 128 // world.size()
        )
        self.assertTrue(tk.allclose(l1, l2))
        self.assertTrue(
            tk.allclose(
                g1["layers"][0]["weight"][part],
                g2["layers"][0]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            tk.allclose(
                g1["layers"][2]["weight"][part],
                g2["layers"][2]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            tk.allclose(
                g1["layers"][1]["weight"][:, part],
                g2["layers"][1]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            tk.allclose(
                g1["layers"][3]["weight"][:, part],
                g2["layers"][3]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            tk.allclose(
                g1["layers"][0]["bias"][part],
                g2["layers"][0]["bias"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            tk.allclose(
                g1["layers"][2]["bias"][part],
                g2["layers"][2]["bias"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            tk.allclose(
                g1["layers"][1]["bias"],
                g2["layers"][1]["bias"],
                atol=self.atol,
                rtol=self.rtol,
            )
        )
        self.assertTrue(
            tk.allclose(
                g1["layers"][3]["bias"],
                g2["layers"][3]["bias"],
                atol=self.atol,
                rtol=self.rtol,
            )
        )

    def test_shard_predicate(self):
        tk.random.seed(0xF0F0F0F0)

        class MyConv(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.aggregate = kwargs.pop("aggregate", False)
                self.conv = nn.Conv2d(*args, **kwargs)

            def __call__(self, x):
                x = self.conv(x)
                if self.aggregate:
                    x = tk.distributed.all_sum(x)
                return x

        def sharding(path, weight):
            parts = path.split(".")
            even = int(parts[1]) % 2 == 0
            if even:
                return 0
            else:
                return -1 if parts[-1] != "bias" else None

        mod = nn.Sequential(
            MyConv(3, 128, kernel_size=3),
            MyConv(128, 128, kernel_size=3),
            MyConv(128, 128, kernel_size=3),
            MyConv(128, 3, kernel_size=3),
        )
        smod = nn.Sequential(
            MyConv(3, 128, kernel_size=3),
            MyConv(128, 128, kernel_size=3, aggregate=True),
            MyConv(128, 128, kernel_size=3),
            MyConv(128, 3, kernel_size=3, aggregate=True),
        )
        smod.update(mod.parameters())
        shard_inplace(smod, sharding)

        x = tk.random.normal((4, 16, 16, 3))
        y1 = mod(x)
        y2 = smod(x)
        self.assertTrue(tk.allclose(y1, y2, atol=1e-6, rtol=1e-4))

    def test_all_gather(self):
        world = tk.distributed.init()
        dtypes = [
            tk.int8,
            tk.uint8,
            tk.int32,
            tk.uint32,
            tk.float32,
            tk.float16,
            tk.bfloat16,
        ]
        for dt in dtypes:
            x = tk.ones((2, 2, 4), dtype=dt)
            y = tk.distributed.all_gather(x)
            self.assertEqual(y.shape, (world.size() * 2, 2, 4))
            self.assertTrue(tk.all(y == 1))

    def test_clip_grad_norm_sharded(self):
        world = tk.distributed.init()
        N = world.size()

        value = 3.0
        grads_slice = {"a": tk.ones((4, 3)) * value, "b": tk.ones((5,)) * value}
        local_numel = 4 * 3 + 5
        expected_norm = math.sqrt(N * local_numel) * value

        clipped, grad_norm = clip_grad_norm_sharded(
            grads_slice, max_norm=1e9, group=world
        )
        tk.eval(clipped, grad_norm)
        self.assertTrue(
            tk.allclose(
                grad_norm, tk.array(expected_norm), atol=self.atol, rtol=self.rtol
            )
        )
        for k in grads_slice:
            self.assertTrue(
                tk.allclose(clipped[k], grads_slice[k], atol=self.atol, rtol=self.rtol)
            )

        max_norm = 1.0
        clipped, grad_norm = clip_grad_norm_sharded(
            grads_slice, max_norm=max_norm, group=world
        )
        tk.eval(clipped, grad_norm)
        scale = max_norm / (expected_norm + 1e-6)
        for k in grads_slice:
            self.assertTrue(
                tk.allclose(
                    clipped[k], grads_slice[k] * scale, atol=self.atol, rtol=self.rtol
                )
            )

    def test_jaccl_all_gather_factory_validation(self):
        # A custom side-channel factory is only valid with the jaccl backend.
        with self.assertRaises(ValueError):
            tk.distributed.init(
                backend="ring",
                all_gather_factory=lambda rank, size: lambda src, n_bytes: b"",
            )

        # The factory must be callable.
        with self.assertRaises(TypeError):
            tk.distributed.init(backend="jaccl", all_gather_factory="not_callable")
