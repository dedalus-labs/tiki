# Copyright © 2023-2024 Apple Inc.

import math
import os
import unittest

import tiki as tk
import tiki_tests


def rope_orig(x, dims, traditional, base, scale, offset, freqs=None):
    N = x.shape[-2]
    dtype = x.dtype
    half_D = dims // 2
    positions = tk.arange(N, dtype=dtype)
    if isinstance(offset, tk.array) and offset.size > 1:
        expand = tuple(range(1, x.ndim - 1))
        positions = tk.expand_dims(offset, expand) + positions
    else:
        positions = offset + positions
    positions = positions * scale
    if freqs is None:
        inv_freqs = tk.exp(
            -tk.arange(0.0, half_D, dtype=dtype) * (math.log(base) / half_D)
        )
    else:
        inv_freqs = (1 / freqs).astype(x.dtype)
    theta = tk.expand_dims(positions, -1) * inv_freqs
    costheta, sintheta = tk.cos(theta), tk.sin(theta)
    if traditional:
        x1 = x[..., :dims:2]
        x2 = x[..., 1:dims:2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta
        rx = tk.concatenate([rx1[..., None], rx2[..., None]], axis=-1)
        if dims < x.shape[-1]:
            rx = tk.reshape(rx, (*x.shape[:-1], dims))
            rx = tk.concatenate([rx, x[..., dims:]], axis=-1)
        return tk.reshape(rx, x.shape)
    else:
        x1 = x[..., : dims // 2]
        x2 = x[..., dims // 2 : dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta
        if dims < x.shape[-1]:
            rx = tk.concatenate([rx1, rx2, x[..., dims:]], axis=-1)
        else:
            rx = tk.concatenate([rx1, rx2], axis=-1)
        return rx


def rms_norm(x, weight, eps):
    x = x.astype(tk.float32)
    x = x * tk.rsqrt(x.square().mean(-1, keepdims=True) + eps)
    return weight * x.astype(weight.dtype)


def layer_norm(x, weight, bias, eps):
    ot = x.dtype
    x = x.astype(tk.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) * tk.rsqrt(var + eps)
    x = x.astype(ot)
    if weight is not None:
        x = x * weight
    if bias is not None:
        x = x + bias
    return x


class TestFast(tiki_tests.TIKITestCase):
    def test_rope(self):
        T = 4

        # Defaults: dims, dtype, base, scale, offset, traditional
        defaults = (8, tk.float32, 10000.0, 1.0, 0, False)

        # Per dtype absolute tolerance
        tolerances = {tk.float32: 1e-6, tk.float16: 1e-3, tk.bfloat16: 1e-2}

        # Test cases:
        dtypes = [tk.float32, tk.float16, tk.bfloat16]
        bases = [10000.0, 1000000.0]
        scales = [1.0, 2.0]
        offsets = [0, 3, tk.array(3)]
        traditional = [True, False]

        for traditional in [True, False]:
            dims, dtype, _, scale, offset, _ = defaults
            for base in bases:
                x = tk.random.uniform(shape=(2, T, dims)).astype(dtype)
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = tk.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

            dims, _, base, scale, offset, _ = defaults
            for dtype in dtypes:
                x = tk.random.uniform(shape=(2, T, dims)).astype(dtype)
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = tk.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                if dtype != tk.float32:
                    ry = rope_orig(
                        x.astype(tk.float32), dims, traditional, base, scale, offset
                    )
                    self.assertLess(tk.abs(ry - rx_fast).max(), tolerances[dtype])
                self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

            dims, dtype, base, scale, _, _ = defaults
            for offset in offsets:
                x = tk.random.uniform(shape=(2, T, dims)).astype(dtype)
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = tk.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

            dims, dtype, base, _, offset, _ = defaults
            for scale in scales:
                x = tk.random.uniform(shape=(2, T, dims)).astype(dtype)
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = tk.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

        # Test transpose into rope
        dims, _, base, scale, offset, traditional = defaults
        x = tk.random.uniform(shape=(1, 1, 4, dims)).swapaxes(1, 2)
        rx = rope_orig(x, dims, traditional, base, scale, offset)
        rx_fast = tk.fast.rope(
            1.0 * x,  # multiply here to allow donation
            dims,
            traditional=traditional,
            base=base,
            scale=scale,
            offset=offset,
        )
        self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[tk.float32])

        # Test raises with integer inputs
        dims, _, base, scale, offset, traditional = defaults
        x = (tk.random.uniform(shape=(2, T, dims)) * 10).astype(tk.int32)
        with self.assertRaises(ValueError):
            y = tk.fast.rope(
                x, dims, traditional=traditional, base=base, scale=scale, offset=offset
            )

    @unittest.skipIf("CI" in os.environ, "Allocates too much memory for CI")
    def test_rope_large_input(self):
        dims, seq_len, batch_size, n_heads = 32, 8192, 8, 32
        base, scale, offset, traditional = 10000.0, 1.0, 0, False
        x = tk.random.normal(shape=[batch_size, seq_len, n_heads, dims]).astype(
            tk.float32
        )
        x = x.swapaxes(1, 2)
        rx_fast = tk.fast.rope(
            x, dims, traditional=traditional, base=base, scale=scale, offset=offset
        )
        ref = rope_orig(x, dims, traditional, base, scale, offset)
        self.assertLess(tk.abs(ref - rx_fast).max(), 5e-3)

    def test_rope_dims_validation(self):
        T = 4
        feature_dim = 64
        x = tk.random.uniform(shape=(1, T, feature_dim))

        # dims = 0 should raise
        with self.assertRaises(ValueError):
            tk.fast.rope(
                x, dims=0, traditional=False, base=10000.0, scale=1.0, offset=0
            )

        # negative dims should raise
        with self.assertRaises(ValueError):
            tk.fast.rope(
                x, dims=-2, traditional=False, base=10000.0, scale=1.0, offset=0
            )

        # odd dims should raise
        with self.assertRaises(ValueError):
            tk.fast.rope(
                x, dims=7, traditional=False, base=10000.0, scale=1.0, offset=0
            )

        # dims > feature_dim should raise
        with self.assertRaises(ValueError):
            tk.fast.rope(
                x, dims=128, traditional=False, base=10000.0, scale=1.0, offset=0
            )

        # valid dims should not raise
        tk.fast.rope(x, dims=32, traditional=False, base=10000.0, scale=1.0, offset=0)
        tk.fast.rope(
            x, dims=feature_dim, traditional=False, base=10000.0, scale=1.0, offset=0
        )

    def test_rope_with_freqs(self):
        tk.random.seed(0)

        # Check throws
        T = 4
        dims = 8
        x = tk.random.uniform(shape=(2, T, dims))

        with self.assertRaises(ValueError):
            freqs = tk.random.uniform(shape=(dims - 1,))
            tk.fast.rope(
                x,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=0,
                freqs=freqs,
            )
        with self.assertRaises(ValueError):
            freqs = tk.random.uniform(shape=(1, dims))
            tk.fast.rope(
                x,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=0,
                freqs=freqs,
            )

        freqs = tk.random.uniform(shape=(dims // 2,))

        tolerances = {tk.float32: 1e-5, tk.float16: 1e-2}
        for dtype in [tk.float32, tk.float16]:
            x_ = x.astype(dtype)
            rx = rope_orig(x_, dims, False, None, 1.0, 0, freqs)
            rx_fast = tk.fast.rope(
                x_,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=0,
                freqs=freqs,
            )
            self.assertEqual(dtype, rx.dtype)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            return

        # Test single vector
        x = tk.random.uniform(shape=(1, 1, dims))
        rx = rope_orig(x, dims, False, None, 1.0, 0, freqs)
        rx_fast = tk.fast.rope(
            x,
            dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=0,
            freqs=freqs,
        )
        self.assertLess(tk.abs(rx - rx_fast).max(), 1e-5)

        # Test grad with freqs
        f1 = lambda x, y: (rope_orig(x, dims, False, None, 1.0, 0, freqs) * y).sum()
        f2 = lambda x, y: (
            tk.fast.rope(
                x,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=0,
                freqs=freqs,
            )
            * y
        ).sum()

        x = tk.random.uniform(shape=(2, 4, dims))
        y = tk.random.uniform(shape=(2, 4, dims))
        g1 = tk.grad(f1)(x, y)
        g2 = tk.grad(f2)(x, y)
        self.assertLess(tk.abs(g1 - g2).max(), 1e-5)

    def test_rope_grad(self):
        D = 32
        defaults = (D, 10000.0, 1.0, 0, False)
        for dims in (D, D // 2):
            for traditional in (True, False):
                _, base, scale, offset, _ = defaults
                f1 = lambda x, y: (
                    rope_orig(x, dims, traditional, base, scale, offset) * y
                ).sum()
                f2 = lambda x, y: (
                    tk.fast.rope(
                        x,
                        dims,
                        traditional=traditional,
                        base=base,
                        scale=scale,
                        offset=offset,
                    )
                    * y
                ).sum()

                x = tk.random.uniform(shape=(2, 100, D))
                y = tk.random.uniform(shape=(2, 100, D))
                g1 = tk.grad(f1)(x, y)
                g2 = tk.grad(f2)(x, y)
                self.assertLess(tk.abs(g1 - g2).max(), 1e-5)

    def test_rope_batch(self):
        T = 4
        base = 10000.0
        scale = 1.0
        traditional = True
        batch_sizes = [3, 8, 11]
        num_heads = [1, 3, 5]
        dims = 32

        x = tk.random.uniform(shape=(8, 4, T, dims))

        offset = tk.array([1, 2, 3])
        with self.assertRaises(ValueError):
            tk.fast.rope(
                x,
                dims,
                traditional=traditional,
                base=base,
                scale=scale,
                offset=offset,
            )

        for batch_size in batch_sizes:
            for n_head in num_heads:
                x = tk.random.uniform(shape=(batch_size, n_head, T, dims))
                offset = tk.arange(batch_size)
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = tk.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                self.assertLess(tk.abs(rx - rx_fast).max(), 1e-5)
        x = tk.random.normal(shape=(2, 6, 8, 64)).transpose(0, 2, 1, 3)
        dims = 64
        offset = 0
        rx_fast = tk.fast.rope(
            x, dims, traditional=traditional, scale=scale, base=base, offset=offset
        )
        rx_fast_single = tk.fast.rope(
            x[0:1], dims, traditional=traditional, scale=scale, base=base, offset=offset
        )

        rx = rope_orig(x, dims, traditional, base, scale, offset)
        self.assertLess(tk.abs(rx - rx_fast).max(), 1e-5)

    def test_rope_single_batch(self):
        base = 10000.0
        scale = 1.0
        offset = 5

        for traditional in [True, False]:
            for B in [2, 4, 8]:
                for n_head in [1, 4, 7]:
                    for dims in [64, 128]:
                        x = tk.random.uniform(shape=(B, n_head, 1, dims))
                        tk.eval(x)
                        rx_fast = tk.fast.rope(
                            x,
                            dims,
                            traditional=traditional,
                            base=base,
                            scale=scale,
                            offset=offset,
                        )
                        rx = rope_orig(x, dims, traditional, base, scale, offset)
                        self.assertLess(tk.abs(rx - rx_fast).max(), 1e-5)

    def test_rope_with_large_offset(self):
        x = tk.random.normal(shape=(1, 1, 1024, 32))
        rx_fp32 = tk.fast.rope(
            x,
            32,
            traditional=False,
            scale=1.0,
            base=10000,
            offset=4000,
        )
        rx_bf16 = tk.fast.rope(
            x.astype(tk.bfloat16),
            32,
            traditional=False,
            scale=1.0,
            base=10000,
            offset=4000,
        )
        self.assertLess((rx_fp32 - rx_bf16).abs().max(), 1e-1)

    def test_rms_norm(self):
        # Per dtype absolute tolerance
        tolerances = {tk.float32: 1e-6, tk.float16: 1e-3, tk.bfloat16: 1e-2}

        dtypes = [tk.float32, tk.float16, tk.bfloat16]
        epss = [1e-3, 1e-5]
        dimss = [31, 32, 33, 256, 512]
        defaults = (tk.float32, 1e-5, 32)

        for dtype in dtypes:
            _, eps, dims = defaults
            x = tk.random.uniform(
                shape=(
                    2,
                    dims,
                )
            ).astype(dtype)
            weight = tk.random.uniform(shape=(dims,)).astype(dtype)
            rx = rms_norm(x, weight, eps)
            rx_fast = tk.fast.rms_norm(x, weight, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = rms_norm(x, tk.ones_like(weight), eps)
            rx_fast = tk.fast.rms_norm(x, None, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

        for eps in epss:
            dtype, _, dims = defaults
            x = tk.random.uniform(shape=(2, dims)).astype(dtype)
            weight = tk.random.uniform(shape=(dims,)).astype(dtype)
            rx = rms_norm(x, weight, eps)
            rx_fast = tk.fast.rms_norm(x, weight, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = rms_norm(x, tk.ones_like(weight), eps)
            rx_fast = tk.fast.rms_norm(x, None, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

        for dims in dimss:
            dtype, eps, _ = defaults
            x = tk.random.uniform(shape=(2, dims)).astype(dtype)
            weight = tk.random.uniform(shape=(dims,)).astype(dtype)
            rx = rms_norm(x, weight, eps)
            rx_fast = tk.fast.rms_norm(x, weight, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = rms_norm(x, tk.ones_like(weight), eps)
            rx_fast = tk.fast.rms_norm(x, None, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

        # Test > 4096
        dims, dtype, eps = 4099, tk.float32, 1e-5
        x = tk.random.uniform(shape=(dims,)).astype(dtype)
        weight = tk.random.uniform(shape=(dims,)).astype(dtype)
        rx = rms_norm(x, weight, eps)
        rx_fast = tk.fast.rms_norm(x, weight, eps)
        self.assertLess(tk.abs(rx - rx_fast).max(), 1e-6)

        # Wrong size w raises
        with self.assertRaises(ValueError):
            x = tk.random.uniform(shape=(1, 5))
            tk.fast.rms_norm(x, tk.ones((4,)), 1e-5)

    def test_rms_norm_grad(self):
        eps = 1e-5
        f1 = lambda x, w, y: (rms_norm(x, w, eps) * y).sum()
        f2 = lambda x, w, y: (tk.fast.rms_norm(x, w, eps) * y).sum()
        f3 = lambda x, y: (rms_norm(x, tk.ones((x.shape[-1],)), eps) * y).sum()
        f4 = lambda x, y: (tk.fast.rms_norm(x, None, eps) * y).sum()

        for D in [32, 256]:
            x = tk.random.uniform(shape=(8, 100, D))
            w = tk.random.uniform(shape=(D,))
            y = tk.random.uniform(shape=(8, 100, D))
            gx1, gw1 = tk.grad(f1, argnums=(0, 1))(x, w, y)
            gx2, gw2 = tk.grad(f2, argnums=(0, 1))(x, w, y)
            self.assertLess(tk.abs(gx1 - gx2).max(), 1e-5)
            self.assertLess(tk.abs(gw1 - gw2).max() / tk.abs(gw1).mean(), 1e-5)
            gx1 = tk.grad(f3, argnums=(0,))(x, y)
            gx2 = tk.grad(f4, argnums=(0,))(x, y)
            self.assertLess(tk.abs(gx1 - gx2).max(), 1e-5)

        D = 8192
        x = tk.random.uniform(shape=(2, 2, D))
        w = tk.random.uniform(shape=(D,))
        y = tk.random.uniform(shape=(2, 2, D))
        gx1, gw1 = tk.grad(f1, argnums=(0, 1))(x, w, y)
        gx2, gw2 = tk.grad(f2, argnums=(0, 1))(x, w, y)
        self.assertLess(tk.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(tk.abs(gw1 - gw2).max() / tk.abs(gw1).mean(), 1e-5)
        gx1 = tk.grad(f3, argnums=(0,))(x, y)
        gx2 = tk.grad(f4, argnums=(0,))(x, y)
        self.assertLess(tk.abs(gx1 - gx2).max(), 1e-5)

        def gf(f):
            def inner(x, w, y):
                gx, gw = tk.grad(f, argnums=(0, 1))(x, w, y)
                return (gx + gw).sum()

            return inner

        gx1, gw1 = tk.grad(gf(f1), argnums=(0, 1))(x, w, y)
        gx2, gw2 = tk.grad(gf(f2), argnums=(0, 1))(x, w, y)
        self.assertLess(tk.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(tk.abs(gw1 - gw2).max() / tk.abs(gw1).mean(), 1e-5)

    def test_cross_entropy(self):
        def cross_entropy_ref(logits, targets):
            score = tk.take_along_axis(logits, tk.expand_dims(targets, -1), -1).squeeze(
                -1
            )
            return tk.logsumexp(logits.astype(tk.float32), axis=-1) - score.astype(
                tk.float32
            )

        tolerances = {tk.float32: 1e-5, tk.float16: 3e-2, tk.bfloat16: 3e-1}

        for V in [7, 32, 128, 255, 256, 1000, 4096, 8192]:
            for dtype in [tk.float32, tk.float16, tk.bfloat16]:
                logits = (tk.random.normal(shape=(4, 7, V), scale=3.0) * 2).astype(
                    dtype
                )
                targets = tk.random.randint(0, V, shape=(4, 7))
                expected = cross_entropy_ref(logits, targets)
                out = tk.fast.cross_entropy(logits, targets)
                self.assertEqual(out.dtype, tk.float32)
                self.assertEqual(out.shape, targets.shape)
                self.assertLess(tk.abs(out - expected).max().item(), tolerances[dtype])

    def test_cross_entropy_shape_checks(self):
        logits = tk.random.normal(shape=(4, 16))
        with self.assertRaises(ValueError):
            tk.fast.cross_entropy(logits, tk.zeros((5,), tk.int32))
        with self.assertRaises(ValueError):
            # Probability targets are not supported by the fused op.
            tk.fast.cross_entropy(logits, tk.zeros((4, 16), tk.int32))
        with self.assertRaises(ValueError):
            tk.fast.cross_entropy(logits, tk.zeros((4,), tk.float32))

    def test_cross_entropy_grad(self):
        def ref(logits, targets):
            score = tk.take_along_axis(logits, tk.expand_dims(targets, -1), -1).squeeze(
                -1
            )
            return tk.logsumexp(logits, axis=-1) - score

        f1 = lambda x, y: ref(x, y).mean()
        f2 = lambda x, y: tk.fast.cross_entropy(x, y).mean()

        for V in [7, 128, 1000, 4096]:
            logits = tk.random.normal(shape=(4, 7, V), scale=2.0)
            targets = tk.random.randint(0, V, shape=(4, 7))
            g1 = tk.grad(f1, argnums=0)(logits, targets)
            g2 = tk.grad(f2, argnums=0)(logits, targets)
            self.assertEqual(g2.shape, logits.shape)
            self.assertLess(tk.abs(g1 - g2).max().item(), 1e-6)

        w = tk.random.uniform(shape=(4, 7))
        f3 = lambda x, y: (ref(x, y) * w).sum()
        f4 = lambda x, y: (tk.fast.cross_entropy(x, y) * w).sum()
        logits = tk.random.normal(shape=(4, 7, 512), scale=2.0)
        targets = tk.random.randint(0, 512, shape=(4, 7))
        g1 = tk.grad(f3, argnums=0)(logits, targets)
        g2 = tk.grad(f4, argnums=0)(logits, targets)
        self.assertEqual(g2.shape, logits.shape)
        self.assertLess(tk.abs(g1 - g2).max().item(), 1e-6)

    def test_layer_norm_dim_check(self):
        with self.assertRaises(ValueError):
            weight = tk.ones((129,))
            x = tk.random.randint(low=0, high=10, shape=(4, 128))
            tk.fast.layer_norm(x, weight, None, 1e-3)

        with self.assertRaises(ValueError):
            bias = tk.ones((129,))
            x = tk.random.randint(low=0, high=10, shape=(4, 128))
            tk.fast.layer_norm(x, None, bias, 1e-3)

    def test_layer_norm(self):
        # Per dtype absolute tolerance
        tolerances = {tk.float32: 1e-5, tk.float16: 5e-3, tk.bfloat16: 5e-2}

        dtypes = [tk.float32, tk.float16, tk.bfloat16]
        epss = [1e-3, 1e-5]
        dimss = [31, 32, 33]
        defaults = (tk.float32, 1e-5, 32)

        for dtype in dtypes:
            _, eps, dims = defaults
            x = tk.random.uniform(
                shape=(
                    2,
                    dims,
                )
            ).astype(dtype)
            weight = tk.random.uniform(shape=(dims,)).astype(dtype)
            bias = tk.random.uniform(shape=(dims,)).astype(dtype)
            rx = layer_norm(x, weight, bias, eps)
            rx_fast = tk.fast.layer_norm(x, weight, bias, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, weight, None, eps)
            rx_fast = tk.fast.layer_norm(x, weight, None, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, bias, eps)
            rx_fast = tk.fast.layer_norm(x, None, bias, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, None, eps)
            rx_fast = tk.fast.layer_norm(x, None, None, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

        for eps in epss:
            dtype, _, dims = defaults
            x = tk.random.uniform(shape=(2, dims)).astype(dtype)
            weight = tk.random.uniform(shape=(dims,)).astype(dtype)
            bias = tk.random.uniform(shape=(dims,)).astype(dtype)
            rx = layer_norm(x, weight, bias, eps)
            rx_fast = tk.fast.layer_norm(x, weight, bias, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, weight, None, eps)
            rx_fast = tk.fast.layer_norm(x, weight, None, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, bias, eps)
            rx_fast = tk.fast.layer_norm(x, None, bias, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, None, eps)
            rx_fast = tk.fast.layer_norm(x, None, None, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

        for dims in dimss:
            dtype, eps, _ = defaults
            x = tk.random.uniform(shape=(2, dims)).astype(dtype)
            weight = tk.random.uniform(shape=(dims,)).astype(dtype)
            bias = tk.random.uniform(shape=(dims,)).astype(dtype)
            rx = layer_norm(x, weight, bias, eps)
            rx_fast = tk.fast.layer_norm(x, weight, bias, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, weight, None, eps)
            rx_fast = tk.fast.layer_norm(x, weight, None, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, bias, eps)
            rx_fast = tk.fast.layer_norm(x, None, bias, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, None, eps)
            rx_fast = tk.fast.layer_norm(x, None, None, eps)
            self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

        # Test > 4096
        dims, dtype, eps = 4099, tk.float32, 1e-5
        x = tk.random.uniform(shape=(dims,)).astype(dtype)
        weight = tk.random.uniform(shape=(dims,)).astype(dtype)
        bias = tk.random.uniform(shape=(dims,)).astype(dtype)
        rx = layer_norm(x, weight, bias, eps)
        rx_fast = tk.fast.layer_norm(x, weight, bias, eps)
        self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
        rx = layer_norm(x, weight, None, eps)
        rx_fast = tk.fast.layer_norm(x, weight, None, eps)
        self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
        rx = layer_norm(x, None, bias, eps)
        rx_fast = tk.fast.layer_norm(x, None, bias, eps)
        self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])
        rx = layer_norm(x, None, None, eps)
        rx_fast = tk.fast.layer_norm(x, None, None, eps)
        self.assertLess(tk.abs(rx - rx_fast).max(), tolerances[dtype])

    def test_slice_into_layer_norm(self):
        dim = 128
        eps = 1e-5
        x = tk.random.uniform(shape=(8, 100, 128))[:, 99:]
        rx_fast = tk.fast.layer_norm(x, weight=None, bias=None, eps=eps)
        rx = layer_norm(x, None, None, eps)
        self.assertLess(tk.abs(rx - rx_fast).max(), 1e-4)

    def test_layer_norm_grad(self):
        D = 32
        eps = 1e-5
        f1 = lambda x, w, b, y: (layer_norm(x, w, b, eps) * y).sum()
        f2 = lambda x, w, b, y: (tk.fast.layer_norm(x, w, b, eps) * y).sum()

        x = tk.random.uniform(shape=(8, 100, D))
        w = tk.random.uniform(shape=(D,))
        b = tk.random.uniform(shape=(D,))
        y = tk.random.uniform(shape=(8, 100, D))

        gx1, gw1, gb1 = tk.grad(f1, argnums=(0, 1, 2))(x, w, b, y)
        gx2, gw2, gb2 = tk.grad(f2, argnums=(0, 1, 2))(x, w, b, y)
        self.assertLess(tk.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(tk.abs(gw1 - gw2).max() / tk.abs(gw1).mean(), 1e-5)
        self.assertLess(tk.abs(gb1 - gb2).max() / tk.abs(gb1).mean(), 1e-5)

        D = 8192
        x = tk.random.uniform(shape=(8, 100, D))
        w = tk.random.uniform(shape=(D,))
        b = tk.random.uniform(shape=(D,))
        y = tk.random.uniform(shape=(8, 100, D))

        gx1, gw1, gb1 = tk.grad(f1, argnums=(0, 1, 2))(x, w, b, y)
        gx2, gw2, gb2 = tk.grad(f2, argnums=(0, 1, 2))(x, w, b, y)
        self.assertLess(tk.abs(gx1 - gx2).max(), 5e-5)
        self.assertLess(tk.abs(gw1 - gw2).max() / tk.abs(gw1).mean(), 5e-5)
        self.assertLess(tk.abs(gb1 - gb2).max() / tk.abs(gb1).mean(), 5e-5)

        def gf(f):
            def inner(x, w, b, y):
                gx, gw, gb = tk.grad(f, argnums=(0, 1, 2))(x, w, b, y)
                return ((gx + gw + gb) * y).sum()

            return inner

        gx1, gw1, gb1 = tk.grad(gf(f1), argnums=(0, 1, 2))(x, w, b, y)
        gx2, gw2, gb2 = tk.grad(gf(f2), argnums=(0, 1, 2))(x, w, b, y)
        self.assertLess(tk.abs(gx1 - gx2).max() / tk.abs(gx1).mean(), 5e-5)
        self.assertLess(tk.abs(gw1 - gw2).max() / tk.abs(gw1).mean(), 5e-5)
        self.assertLess(tk.abs(gb1).max(), 1e-9)
        self.assertLess(tk.abs(gb2).max(), 1e-9)

    def test_layer_norm_grad_no_bias(self):
        # Second-order gradient through layer_norm with weight but no bias.
        # Regression test: the VJP fallback had zeros_like(w) instead of
        # zeros_like(b) for the bias placeholder gradient, causing a shape
        # mismatch that crashes on higher-order differentiation.
        D = 8
        eps = 1e-5
        x = tk.random.uniform(shape=(2, 4, D))
        w = tk.random.uniform(shape=(D,))
        y = tk.random.uniform(shape=(2, 4, D))
        tk.eval(x, w, y)

        f_ref = lambda x, w, y: (layer_norm(x, w, None, eps) * y).sum()
        f_fast = lambda x, w, y: (tk.fast.layer_norm(x, w, None, eps) * y).sum()

        # First order should match reference
        gx1, gw1 = tk.grad(f_ref, argnums=(0, 1))(x, w, y)
        gx2, gw2 = tk.grad(f_fast, argnums=(0, 1))(x, w, y)
        self.assertLess(tk.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(tk.abs(gw1 - gw2).max() / tk.abs(gw1).mean(), 1e-5)

        # Second order — this crashes without the fix due to shape mismatch
        # in the bias placeholder gradient: zeros_like(w) shape (D,) vs
        # expected zeros_like(b) shape ()
        def gf(f):
            def inner(x, w, y):
                gx, gw = tk.grad(f, argnums=(0, 1))(x, w, y)
                return ((gx + gw) * y).sum()

            return inner

        gx1, gw1 = tk.grad(gf(f_ref), argnums=(0, 1))(x, w, y)
        gx2, gw2 = tk.grad(gf(f_fast), argnums=(0, 1))(x, w, y)
        self.assertLess(tk.abs(gx1 - gx2).max() / tk.abs(gx1).mean(), 5e-5)
        self.assertLess(tk.abs(gw1 - gw2).max() / tk.abs(gw1).mean(), 5e-5)

    def test_layer_norm_grad_no_params(self):
        eps = 1e-5
        f1 = lambda x: layer_norm(x, None, None, eps).sum()
        f2 = lambda x: tk.fast.layer_norm(x, None, None, eps).sum()
        x = tk.random.normal(shape=(2, 2, 8))
        tk.eval(x)

        gx1 = tk.grad(f1)(x)
        gx2 = tk.grad(f2)(x)
        self.assertTrue(tk.allclose(gx1, gx2, atol=1e-6))

    def test_layer_norm_grad_params(self):
        eps = 1e-5
        f1 = lambda params, x: (layer_norm(x, params[0], params[1], eps)).sum()
        f2 = lambda params, x: (tk.fast.layer_norm(x, params[0], params[1], eps)).sum()

        w = tk.ones((8,))
        b = tk.zeros((8,))
        x = tk.random.normal(shape=(2, 2, 8))
        tk.eval(x, w, b)

        gw1, gb1 = tk.grad(f1)((w, b), x)
        gw2, gb2 = tk.grad(f2)((w, b), x)
        self.assertLess(tk.abs(gw1 - gw2).max() / tk.abs(gw1).mean(), 1e-5)
        self.assertLess(tk.abs(gb1 - gb2).max() / tk.abs(gb1).mean(), 1e-5)

    def test_fast_transforms(self):
        x = tk.random.uniform(shape=(2, 2, 8))

        defaults = (8, False, 10000.0, 1.0, 0)
        dims, traditional, base, scale, offset = defaults

        # VJP
        _, vjp_out = tk.vjp(lambda x: rope_orig(x, *defaults), (x,), (tk.ones_like(x),))
        _, vjp_fast_out = tk.vjp(
            lambda x: tk.fast.rope(
                x, dims, traditional=traditional, base=base, scale=scale, offset=offset
            ),
            (x,),
            (tk.ones_like(x),),
        )
        self.assertTrue(tk.allclose(vjp_out[0], vjp_fast_out[0]))

        # JVP
        _, jvp_out = tk.jvp(lambda x: rope_orig(x, *defaults), (x,), (tk.ones_like(x),))
        _, jvp_fast_out = tk.jvp(
            lambda x: tk.fast.rope(
                x, dims, traditional=traditional, base=base, scale=scale, offset=offset
            ),
            (x,),
            (tk.ones_like(x),),
        )
        self.assertTrue(tk.allclose(jvp_out[0], jvp_fast_out[0]))

        # VMAP
        x = tk.random.uniform(shape=(2, 2, 2, 8))
        vmap_out = tk.vmap(lambda x: rope_orig(x, *defaults))(x)
        vmap_fast_out = tk.vmap(
            lambda x: tk.fast.rope(
                x, dims, traditional=traditional, base=base, scale=scale, offset=offset
            )
        )(x)
        self.assertTrue(tk.allclose(vmap_out, vmap_fast_out))

    @unittest.skipIf(not tk.is_available(tk.gpu), "No GPU available")
    def test_custom_kernel_basic(self):
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

        tk.random.seed(7)
        a = tk.random.normal(shape=(2, 2))
        kernel = custom_kernel(
            name="basic",
            input_names=["a"],
            output_names=["out1"],
            source=source,
        )
        out = kernel(
            inputs=[a],
            grid=(4, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(2, 2)],
            output_dtypes=[tk.float32],
            stream=tk.gpu,
        )
        self.assertTrue(tk.allclose(out[0], a))

    @unittest.skipIf(not tk.is_available(tk.gpu), "No GPU available")
    def test_custom_kernel_args(self):
        if tk.metal.is_available():
            source = """
                uint elem = thread_position_in_grid.x;
                T tmp = a[0];
                if (e) {
                    out1[elem] = a[1] + b[2] + c[3] + d + f;
                } else {
                    out1[elem] = 1;
                }
                out2[elem] = a[1] + b[2] + c[1] - d;
            """
            custom_kernel = tk.fast.metal_kernel
        elif tk.cuda.is_available():
            source = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                T tmp = a[0];
                if (e) {
                    out1[elem] = a[1] + b[2] + static_cast<float>(c[3]) + d[0] + f;
                } else {
                    out1[elem] = 1;
                }
                out2[elem] = a[1] + b[2] + static_cast<float>(c[1]) - d[0];
            """
            custom_kernel = tk.fast.cuda_kernel

        tk.random.seed(7)
        a = tk.random.normal(shape=(3, 6))
        c = tk.random.normal(shape=(2, 2)).astype(tk.bfloat16)

        kernel = custom_kernel(
            name="arg_test",
            input_names=["a", "b", "c", "d"],
            output_names=["out1", "out2"],
            source=source,
        )
        out = kernel(
            inputs=[
                a,
                tk.array([3, 4, 5]),
                c,
                7.3,
            ],
            template=[
                ("e", True),
                ("f", 3),
                ("T", tk.float16),
            ],
            grid=(6, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(3, 2), (3, 2)],
            output_dtypes=[tk.float32, tk.int32],
            stream=tk.gpu,
        )

        self.assertTrue(tk.allclose(out[0], tk.full((3, 2), 14.0484)))
        self.assertTrue(tk.allclose(out[1], tk.full((3, 2), -2, dtype=tk.int32)))

    @unittest.skipIf(not tk.is_available(tk.gpu), "No GPU available")
    def test_custom_kernel_strides(self):
        if tk.metal.is_available():
            source = """
                uint elem = thread_position_in_grid.x;
                uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
                T tmp = inp[loc];
                out[elem] = metal::precise::exp(tmp) * threads_per_simdgroup;
            """
            source_contig = """
                uint elem = thread_position_in_grid.x;
                T tmp = inp[elem];
                out[elem] = metal::precise::exp(tmp) * threads_per_simdgroup;
            """
            custom_kernel = tk.fast.metal_kernel
        elif tk.cuda.is_available():
            source = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                auto loc = elem_to_loc(elem, inp_shape.data(), inp_strides.data(), inp_ndim);
                T tmp = inp[loc];
                out[elem] = exp(tmp) * WARP_SIZE;
            """
            source_contig = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                T tmp = inp[elem];
                out[elem] = exp(tmp) * WARP_SIZE;
            """
            custom_kernel = tk.fast.cuda_kernel

        tk.random.seed(7)
        a = tk.random.normal(shape=(3, 6))

        # non contiguous
        a = tk.tile(a[::2], [4, 1])

        for contig in [True, False]:
            kernel = custom_kernel(
                name="myexp" + str(contig),
                input_names=["inp"],
                output_names=["out"],
                source=source_contig if contig else source,
                ensure_row_contiguous=contig,
            )
            outputs = kernel(
                inputs=[a],
                template=[("T", tk.float32)],
                grid=(a.size, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=tk.gpu,
            )
            self.assertTrue(tk.allclose(tk.exp(a) * 32, outputs[0]))

    @unittest.skipIf(not tk.is_available(tk.gpu), "No GPU available")
    def test_custom_kernel_helper(self):
        if tk.metal.is_available():
            header = """
            template <typename T>
            T do_exp(T x) {
                return metal::precise::exp(x);
            }
            """
            source = """
                uint elem = thread_position_in_grid.x;
                out1[elem] = do_exp(a[elem]);
            """
            custom_kernel = tk.fast.metal_kernel
        elif tk.cuda.is_available():
            header = """
            template <typename T>
            __device__ T do_exp(T x) {
                return exp(x);
            }
            """
            source = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                out1[elem] = do_exp(a[elem]);
            """
            custom_kernel = tk.fast.cuda_kernel

        tk.random.seed(7)
        a = tk.random.normal(shape=(2, 2))
        kernel = custom_kernel(
            name="helper",
            input_names=["a"],
            output_names=["out1"],
            header=header,
            source=source,
        )
        out = kernel(
            inputs=[a],
            grid=(4, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(2, 2)],
            output_dtypes=[tk.float32],
            stream=tk.gpu,
        )
        self.assertTrue(tk.allclose(out[0], tk.exp(a)))

    @unittest.skipIf(not tk.is_available(tk.gpu), "No GPU available")
    def test_custom_kernel_attributes(self):
        if tk.metal.is_available():
            source = "out[0] = threads_per_threadgroup.x;"
            custom_kernel = tk.fast.metal_kernel
        elif tk.cuda.is_available():
            source = "out[0] = blockDim.x;"
            custom_kernel = tk.fast.cuda_kernel

        a = tk.zeros(shape=(1, 1))
        kernel = custom_kernel(
            name="test_fun",
            input_names=["a"],
            output_names=["out"],
            source=source,
        )
        out = kernel(
            inputs=[a],
            grid=(2, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(1, 1)],
            output_dtypes=[tk.uint32],
            stream=tk.gpu,
        )[0]
        self.assertEqual(out.item(), 2)

    @unittest.skipIf(not tk.metal.is_available(), "Metal is not available")
    def test_custom_kernel_caching(self):
        def call_kernel(a: tk.array, source):
            kernel = tk.fast.metal_kernel(
                name="my_kernel",
                input_names=["inp"],
                output_names=["out"],
                source=source,
            )
            return kernel(
                inputs=[a],
                grid=(a.size, 1, 1),
                threadgroup=(a.size, 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=tk.gpu,
            )[0]

        a = tk.random.normal(shape=(32,))

        source = """
            uint elem = thread_position_in_grid.x;
            out[elem] = 0.0;
        """

        out = call_kernel(a, source)
        self.assertTrue(tk.array_equal(out, tk.zeros_like(out)))

        source = """
            uint elem = thread_position_in_grid.x;
            out[elem] = 1.0;
        """
        out = call_kernel(a, source)
        self.assertTrue(tk.array_equal(out, tk.ones_like(out)))

    @unittest.skipIf(not tk.metal.is_available(), "Metal is not available")
    def test_custom_kernel_same_name_different_source_one_eval(self):
        # Regression test for #3832: two kernels sharing a name but with
        # different sources, dispatched in a SINGLE eval batch, must each run
        # their own compiled code instead of silently reusing the first's.
        def call_kernel(a, source):
            kernel = tk.fast.metal_kernel(
                name="dup_name",
                input_names=["inp"],
                output_names=["out"],
                source=source,
            )
            return kernel(
                inputs=[a],
                grid=(a.size, 1, 1),
                threadgroup=(a.size, 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=tk.gpu,
            )[0]

        a = tk.arange(32, dtype=tk.float32)
        out_a = call_kernel(
            a, "uint e = thread_position_in_grid.x; out[e] = inp[e] * 2.0f;"
        )
        out_b = call_kernel(
            a, "uint e = thread_position_in_grid.x; out[e] = inp[e] + 100.0f;"
        )
        tk.eval(out_a, out_b)  # one batch — the reported failure case
        self.assertTrue(tk.array_equal(out_a, a * 2.0))
        self.assertTrue(tk.array_equal(out_b, a + 100.0))

    @unittest.skipIf(not tk.cuda.is_available(), "CUDA is not available")
    def test_cuda_kernel_same_name_different_source(self):
        # The CUDA module cache was keyed on the kernel name alone, so the
        # second kernel here silently ran the first one's code. Metal had the
        # same bug, fixed in #3833.
        def call_kernel(a, source):
            kernel = tk.fast.cuda_kernel(
                name="dup_name",
                input_names=["inp"],
                output_names=["out"],
                source=source,
            )
            return kernel(
                inputs=[a],
                grid=(a.size, 1, 1),
                threadgroup=(a.size, 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=tk.gpu,
            )[0]

        a = tk.arange(32, dtype=tk.float32)
        elem = "auto e = cooperative_groups::this_grid().thread_rank();"
        out_a = call_kernel(a, f"{elem} out[e] = inp[e] * 2.0f;")
        out_b = call_kernel(a, f"{elem} out[e] = inp[e] + 100.0f;")
        tk.eval(out_a, out_b)
        self.assertTrue(tk.array_equal(out_a, a * 2.0))
        self.assertTrue(tk.array_equal(out_b, a + 100.0))

    @unittest.skipIf(not tk.metal.is_available(), "Metal is not available")
    def test_custom_metal_kernel_math_mode(self):
        with self.assertRaises(ValueError):
            tk.fast.metal_kernel(
                name="invalid_math_mode",
                input_names=["inp"],
                output_names=["out"],
                source="out[0] = inp[0];",
                compile_options={"math_mode": "precise"},
            )

        with self.assertRaises(ValueError):
            tk.fast.metal_kernel(
                name="invalid_compile_options",
                input_names=["inp"],
                output_names=["out"],
                source="out[0] = inp[0];",
                compile_options={"unknown": "value"},
            )

        # Numerical special cases such as exp(-inf) can agree between math
        # modes, so they don't reliably detect whether the mode was applied.
        # Branch on the compiler's __FAST_MATH__ macro instead: it is defined
        # only when fast math is enabled, so the test fails if the selected
        # math mode is not forwarded to the Metal compiler.
        source = """
            uint elem = thread_position_in_grid.x;
            #if defined(__FAST_MATH__) && __FAST_MATH__
            out[elem] = 1.0f;
            #else
            out[elem] = 0.0f;
            #endif
        """

        a = tk.zeros((4,), dtype=tk.float32)
        expected = {
            "safe": tk.zeros_like(a),
            "fast": tk.ones_like(a),
        }

        # Reuse the same kernel name across modes so the library cache is forced
        # to rebuild when the math mode changes, guarding against a stale build
        # being returned for a different mode.
        for mode, expected_out in expected.items():
            kernel = tk.fast.metal_kernel(
                name="math_mode",
                input_names=["inp"],
                output_names=["out"],
                source=source,
                compile_options={"math_mode": mode},
            )
            out = kernel(
                inputs=[a],
                grid=(a.size, 1, 1),
                threadgroup=(a.size, 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=tk.gpu,
            )[0]
            self.assertTrue(tk.array_equal(out, expected_out))

    @unittest.skipIf(not tk.metal.is_available(), "Metal is not available")
    def test_custom_kernel_mixed_dtypes(self):
        # Calling the same kernel with different input dtypes in a single
        # graph should not invalidate pipeline states that are still in use
        # by an uncommitted command buffer
        kernel = tk.fast.metal_kernel(
            name="mixed_dtypes",
            input_names=["inp"],
            output_names=["out"],
            source="""
                uint elem = thread_position_in_grid.x;
                out[elem] = inp[elem] + inp[elem];
            """,
        )

        def call_kernel(a: tk.array):
            return kernel(
                inputs=[a],
                grid=(a.size, 1, 1),
                threadgroup=(a.size, 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=tk.gpu,
            )[0]

        a = tk.full((32,), 1.5, dtype=tk.float16)
        b = tk.full((32,), 2.5, dtype=tk.float32)
        out = call_kernel(a).astype(tk.float32) + call_kernel(b)
        self.assertTrue(tk.allclose(out, tk.full((32,), 8.0)))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
