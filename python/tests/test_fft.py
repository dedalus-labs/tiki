# Copyright © 2023 Apple Inc.

import itertools
import unittest

import numpy as np
import tiki as tk
import tiki_tests

try:
    import torch

    has_torch = True
except ImportError as e:
    has_torch = False


class TestFFT(tiki_tests.TIKITestCase):
    def check_mx_np(self, op_mx, op_np, a_np, atol=1e-5, rtol=1e-6, **kwargs):
        out_np = op_np(a_np, **kwargs)
        a_mx = tk.array(a_np)
        out_mx = op_mx(a_mx, **kwargs)
        np.testing.assert_allclose(out_np, out_mx, atol=atol, rtol=rtol)

    def test_fft(self):
        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        self.check_mx_np(tk.fft.fft, np.fft.fft, a_np)

        # Check with slicing and padding
        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        self.check_mx_np(tk.fft.fft, np.fft.fft, a_np, n=80)
        self.check_mx_np(tk.fft.fft, np.fft.fft, a_np, n=120)

        # Check different axes
        r = np.random.rand(100, 100).astype(np.float32)
        i = np.random.rand(100, 100).astype(np.float32)
        a_np = r + 1j * i
        self.check_mx_np(tk.fft.fft, np.fft.fft, a_np, axis=0)
        self.check_mx_np(tk.fft.fft, np.fft.fft, a_np, axis=1)

        # Check real fft
        a_np = np.random.rand(100).astype(np.float32)
        self.check_mx_np(tk.fft.rfft, np.fft.rfft, a_np)
        self.check_mx_np(tk.fft.rfft, np.fft.rfft, a_np, n=80)
        self.check_mx_np(tk.fft.rfft, np.fft.rfft, a_np, n=120)

        # Check real inverse
        r = np.random.rand(100, 100).astype(np.float32)
        i = np.random.rand(100, 100).astype(np.float32)
        a_np = r + 1j * i
        self.check_mx_np(tk.fft.ifft, np.fft.ifft, a_np)
        self.check_mx_np(tk.fft.ifft, np.fft.ifft, a_np, n=80)
        self.check_mx_np(tk.fft.ifft, np.fft.ifft, a_np, n=120)

        x = np.fft.rfft(np.real(a_np))
        self.check_mx_np(tk.fft.irfft, np.fft.irfft, x)

    def test_fftn(self):
        r = np.random.randn(8, 8, 8).astype(np.float32)
        i = np.random.randn(8, 8, 8).astype(np.float32)
        a = r + 1j * i

        axes = [None, (1, 2), (2, 1), (0, 2)]
        shapes = [None, (10, 5), (5, 10)]
        ops = [
            "fft2",
            "ifft2",
            "rfft2",
            "irfft2",
            "fftn",
            "ifftn",
            "rfftn",
            "irfftn",
        ]

        for op, ax, s in itertools.product(ops, axes, shapes):
            if ax is None and s is not None:
                continue
            x = a
            if op in ["rfft2", "rfftn"]:
                x = r
            elif op == "irfft2":
                x = np.ascontiguousarray(np.fft.rfft2(r, axes=ax, s=s))
            elif op == "irfftn":
                x = np.ascontiguousarray(np.fft.rfftn(r, axes=ax, s=s))
            mx_op = getattr(tk.fft, op)
            np_op = getattr(np.fft, op)
            self.check_mx_np(mx_op, np_op, x, axes=ax, s=s)

        # Explicitly exercise transposed layouts and axes that are not
        # physically last in memory order.
        xt = np.transpose(a, (1, 2, 0))
        self.check_mx_np(tk.fft.fftn, np.fft.fftn, xt, axes=(2, 0))
        self.check_mx_np(tk.fft.ifftn, np.fft.ifftn, xt, axes=(2, 0))

        rt = np.transpose(r, (1, 2, 0))
        self.check_mx_np(tk.fft.rfftn, np.fft.rfftn, rt, axes=(2, 0))
        irfft_in = np.ascontiguousarray(np.fft.rfftn(rt, axes=(2, 0)))
        self.check_mx_np(tk.fft.irfftn, np.fft.irfftn, irfft_in, axes=(2, 0))

    def test_fft_norm(self):
        norms = ["backward", "ortho", "forward"]

        r = np.random.randn(8, 6).astype(np.float32)
        i = np.random.randn(8, 6).astype(np.float32)
        c = r + 1j * i

        for norm in norms:
            self.check_mx_np(tk.fft.fft, np.fft.fft, c, axis=1, norm=norm)
            self.check_mx_np(tk.fft.ifft, np.fft.ifft, c, axis=1, norm=norm)
            self.check_mx_np(tk.fft.rfft, np.fft.rfft, r, axis=1, norm=norm)

            cr = np.fft.rfft(r, axis=1)
            self.check_mx_np(tk.fft.irfft, np.fft.irfft, cr, axis=1, norm=norm)

            self.check_mx_np(tk.fft.fft2, np.fft.fft2, c, axes=(0, 1), norm=norm)
            self.check_mx_np(tk.fft.ifft2, np.fft.ifft2, c, axes=(0, 1), norm=norm)
            self.check_mx_np(tk.fft.fftn, np.fft.fftn, c, axes=(0, 1), norm=norm)
            self.check_mx_np(tk.fft.ifftn, np.fft.ifftn, c, axes=(0, 1), norm=norm)

            self.check_mx_np(tk.fft.rfft2, np.fft.rfft2, r, axes=(0, 1), norm=norm)
            self.check_mx_np(tk.fft.rfftn, np.fft.rfftn, r, axes=(0, 1), norm=norm)

            cr2 = np.fft.rfft2(r, axes=(0, 1))
            self.check_mx_np(tk.fft.irfft2, np.fft.irfft2, cr2, axes=(0, 1), norm=norm)
            self.check_mx_np(tk.fft.irfftn, np.fft.irfftn, cr2, axes=(0, 1), norm=norm)

    def _run_ffts(self, shape, atol=1e-4, rtol=1e-4):
        np.random.seed(9)

        r = np.random.rand(*shape).astype(np.float32)
        i = np.random.rand(*shape).astype(np.float32)
        a_np = r + 1j * i
        self.check_mx_np(tk.fft.fft, np.fft.fft, a_np, atol=atol, rtol=rtol)
        self.check_mx_np(tk.fft.ifft, np.fft.ifft, a_np, atol=atol, rtol=rtol)

        self.check_mx_np(tk.fft.rfft, np.fft.rfft, r, atol=atol, rtol=rtol)

        ia_np = np.fft.rfft(r)
        self.check_mx_np(
            tk.fft.irfft, np.fft.irfft, ia_np, atol=atol, rtol=rtol, n=shape[-1]
        )
        self.check_mx_np(tk.fft.irfft, np.fft.irfft, ia_np, atol=atol, rtol=rtol)

    def test_fft_shared_mem(self):
        nums = np.concatenate(
            [
                # small radix
                np.arange(2, 14),
                # powers of 2
                [2**k for k in range(4, 13)],
                # stockham
                [3 * 3 * 3, 3 * 11, 11 * 13 * 2, 7 * 4 * 13 * 11, 13 * 13 * 11],
                # rader
                [17, 23, 29, 17 * 8 * 3, 23 * 2, 1153, 1982],
                # bluestein
                [47, 83, 17 * 17],
                # large stockham
                [3159, 3645, 3969, 4004],
            ]
        )
        for batch_size in (1, 3, 32):
            for num in nums:
                atol = 1e-4 if num < 1025 else 1e-3
                self._run_ffts((batch_size, num), atol=atol)

    @unittest.skipIf(not tk.metal.is_available(), "Metal is not available")
    def test_batched_bluestein_twiddle_table(self):
        for batch, n in ((1023, 1031), (1024, 1031), (1024, 1531)):
            with self.subTest(batch=batch, n=n), tk.stream(tk.gpu):
                index = tk.arange(n, dtype=tk.float32)
                base = tk.cos(index * 0.017) + 0.25 * tk.sin(index * 0.031)
                base = base + 1j * (
                    tk.sin(index * 0.023) - 0.125 * tk.cos(index * 0.047)
                )
                scale_index = tk.arange(batch, dtype=tk.float32)
                scales = (0.75 + 0.25 * tk.cos(scale_index * 0.013)) * (
                    tk.cos(scale_index * 0.019) + 1j * tk.sin(scale_index * 0.019)
                )
                signal = scales[:, None] * base[None, :]

                for transform, atol in ((tk.fft.fft, 1e-3), (tk.fft.ifft, 1e-4)):
                    expected = scales[:, None] * transform(base)[None, :]
                    output = transform(signal)
                    tk.eval(expected, output)
                    self.assertTrue(
                        tk.allclose(output, expected, atol=atol, rtol=1e-4).item()
                    )

    @unittest.skip("Too slow for CI but useful for local testing.")
    def test_fft_exhaustive(self):
        nums = range(2, 4097)
        for batch_size in (1, 3, 32):
            for num in nums:
                print(num)
                atol = 1e-4 if num < 1025 else 1e-3
                self._run_ffts((batch_size, num), atol=atol)

    def test_fft_big_powers_of_two(self):
        # TODO: improve precision on big powers of two on GPU
        for k in range(12, 17):
            self._run_ffts((3, 2**k), atol=1e-3)

        for k in range(17, 20):
            self._run_ffts((3, 2**k), atol=1e-2)

        # Past 2**20 the four step plan has to grow n2 to keep both factors
        # inside threadgroup memory
        for k in range(20, 25):
            self._run_ffts((1, 2**k), atol=1e-2, rtol=1e-3)

    @unittest.skipIf(
        not tk.metal.is_available(), "the size limit is specific to the Metal FFT plan"
    )
    def test_fft_too_large(self):
        # Larger than the four step plan can decompose, so it has to throw
        # rather than run a kernel that silently returns the wrong answer.
        # CUDA hands this to cuFFT instead and has no such limit.
        with self.assertRaises(RuntimeError):
            tk.eval(tk.fft.fft(tk.zeros(2**25)))

    def test_fft_large_numbers(self):
        numbers = [
            1037,  # prime > 2048
            18247,  # medium size prime factors
            1259 * 11,  # large prime factors
            7883,  # large prime
            3**8,  # large stockham decomposable
            3109,  # bluestein
            4006,  # large rader
        ]
        for large_num in numbers:
            self._run_ffts((1, large_num), atol=1e-3)

    def test_fft_contiguity(self):
        r = np.random.rand(4, 8).astype(np.float32)
        i = np.random.rand(4, 8).astype(np.float32)
        a_np = r + 1j * i
        a_mx = tk.array(a_np)

        # non-contiguous in the FFT dim
        out_mx = tk.fft.fft(a_mx[:, ::2])
        out_np = np.fft.fft(a_np[:, ::2])
        np.testing.assert_allclose(out_np, out_mx, atol=1e-5, rtol=1e-5)

        # non-contiguous not in the FFT dim
        out_mx = tk.fft.fft(a_mx[::2])
        out_np = np.fft.fft(a_np[::2])
        np.testing.assert_allclose(out_np, out_mx, atol=1e-5, rtol=1e-5)

        out_mx = tk.broadcast_to(tk.reshape(tk.transpose(a_mx), (4, 8, 1)), (4, 8, 16))
        out_np = np.broadcast_to(np.reshape(np.transpose(a_np), (4, 8, 1)), (4, 8, 16))
        np.testing.assert_allclose(out_np, out_mx, atol=1e-5, rtol=1e-5)

        out2_mx = tk.fft.fft(tk.abs(out_mx) + 4)
        out2_np = np.fft.fft(np.abs(out_np) + 4)
        np.testing.assert_allclose(out2_mx, out2_np, atol=1e-5, rtol=1e-5)

        b_np = np.array([[0, 1, 2, 3]])
        out_mx = tk.abs(tk.fft.fft(tk.tile(tk.reshape(tk.array(b_np), (1, 4)), (4, 1))))
        out_np = np.abs(np.fft.fft(np.tile(np.reshape(np.array(b_np), (1, 4)), (4, 1))))
        np.testing.assert_allclose(out_mx, out_np, atol=1e-5, rtol=1e-5)

    def test_fft_into_ifft(self):
        n_fft = 8193
        tk.random.seed(0)

        segment = tk.random.normal(shape=[1, n_fft]) + 1j * tk.random.normal(
            shape=(1, n_fft)
        )
        segment = tk.fft.fft(segment, n=n_fft)
        r = tk.fft.ifft(segment, n=n_fft)
        r_np = np.fft.ifft(segment, n=n_fft)
        self.assertTrue(np.allclose(r, r_np, atol=1e-5, rtol=1e-5))

    def test_fft_throws(self):
        x = tk.array(3.0)
        with self.assertRaises(ValueError):
            tk.fft.irfftn(x)
        with self.assertRaises(ValueError):
            tk.fft.fft(tk.array([1.0]), norm="invalid")

    def test_fftfreq(self):
        for n, d in [(1, 1.0), (4, 0.5), (5, 0.25), (8, -0.5), (6, 1.0)]:
            out = tk.fft.fftfreq(n, d=d)
            expected = np.fft.fftfreq(n, d=d).astype(np.float32)
            self.assertEqual(out.dtype, tk.float32)
            np.testing.assert_allclose(out, expected, atol=0.0, rtol=0.0)

        with self.assertRaises(ValueError):
            tk.fft.fftfreq(0)

        with self.assertRaises(ValueError):
            tk.fft.fftfreq(-1)

        # Test default d=1.0
        out = tk.fft.fftfreq(8)
        expected = np.fft.fftfreq(8).astype(np.float32)
        np.testing.assert_allclose(out, expected, atol=0.0, rtol=0.0)

        with self.assertRaises(ValueError):
            tk.fft.fftfreq(4, d=0.0)

    def test_rfftfreq(self):
        for n, d in [(1, 1.0), (4, 0.5), (5, 0.25), (8, -0.5), (6, 1.0)]:
            out = tk.fft.rfftfreq(n, d=d)
            expected = np.fft.rfftfreq(n, d=d).astype(np.float32)
            self.assertEqual(out.dtype, tk.float32)
            np.testing.assert_allclose(out, expected, atol=0.0, rtol=0.0)

        # Test default d=1.0
        out = tk.fft.rfftfreq(8)
        expected = np.fft.rfftfreq(8).astype(np.float32)
        np.testing.assert_allclose(out, expected, atol=0.0, rtol=0.0)

        with self.assertRaises(ValueError):
            tk.fft.rfftfreq(0)

        with self.assertRaises(ValueError):
            tk.fft.rfftfreq(-1)

        with self.assertRaises(ValueError):
            tk.fft.rfftfreq(4, d=0.0)

    def test_fftshift(self):
        # Test 1D arrays
        r = np.random.rand(100).astype(np.float32)
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r)

        # Test with specific axis
        r = np.random.rand(4, 6).astype(np.float32)
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r, axes=0)
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r, axes=1)
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r, axes=[0])
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r, axes=[1])
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r, axes=[0, 1])

        # Test with negative axes
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r, axes=-1)
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r, axes=[-1])

        # Test with odd lengths
        r = np.random.rand(5, 7).astype(np.float32)
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r)
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, r, axes=[0])

        # Test with complex input
        r = np.random.rand(8, 8).astype(np.float32)
        i = np.random.rand(8, 8).astype(np.float32)
        c = r + 1j * i
        self.check_mx_np(tk.fft.fftshift, np.fft.fftshift, c)

    def test_ifftshift(self):
        # Test 1D arrays
        r = np.random.rand(100).astype(np.float32)
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r)

        # Test with specific axis
        r = np.random.rand(4, 6).astype(np.float32)
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r, axes=0)
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r, axes=1)
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r, axes=[0])
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r, axes=[1])
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r, axes=[0, 1])

        # Test with negative axes
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r, axes=-1)
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r, axes=[-1])

        # Test with odd lengths
        r = np.random.rand(5, 7).astype(np.float32)
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r)
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, r, axes=[0])

        # Test with complex input
        r = np.random.rand(8, 8).astype(np.float32)
        i = np.random.rand(8, 8).astype(np.float32)
        c = r + 1j * i
        self.check_mx_np(tk.fft.ifftshift, np.fft.ifftshift, c)

    def test_fftshift_errors(self):
        # Test invalid axes
        x = tk.array(np.random.rand(4, 4).astype(np.float32))
        with self.assertRaises(ValueError):
            tk.fft.fftshift(x, axes=[2])
        with self.assertRaises(ValueError):
            tk.fft.fftshift(x, axes=[-3])

        # Test empty array
        x = tk.array([])
        self.assertTrue(tk.array_equal(tk.fft.fftshift(x), x))

    @unittest.skipIf(not has_torch, "requires PyTorch")
    def test_fft_grads(self):
        real = [True, False]
        inverse = [True, False]
        axes = [
            (-1,),
            (-2, -1),
        ]
        shapes = [
            (4, 4),
            (2, 4),
            (2, 7),
            (7, 7),
        ]

        mxffts = {
            (True, True): tk.fft.irfftn,
            (True, False): tk.fft.rfftn,
            (False, True): tk.fft.ifftn,
            (False, False): tk.fft.fftn,
        }
        tffts = {
            (True, True): torch.fft.irfftn,
            (True, False): torch.fft.rfftn,
            (False, True): torch.fft.ifftn,
            (False, False): torch.fft.fftn,
        }

        for r, i, ax, sh in itertools.product(real, inverse, axes, shapes):

            def f(x):
                y = mxffts[r, i](x)
                return (tk.abs(y) ** 2).sum()

            def g(x):
                y = tffts[r, i](x)
                return (torch.abs(y) ** 2).sum()

            if r and not i:
                x = tk.random.normal(sh)
            else:
                x = tk.random.normal((*sh, 2)).view(tk.complex64).squeeze()
            tk.eval(x)
            x_torch = torch.tensor(x, device="cpu")
            fx = f(x)
            gx = g(x_torch)
            self.assertLess((fx - gx).abs().max() / gx.abs().mean(), 1e-4)

            dfdx = tk.grad(f)(x)
            dgdx = torch.func.grad(g)(x_torch)
            self.assertLess((dfdx - dgdx).abs().max() / dgdx.abs().mean(), 1e-4)

    def make_ffts(self):
        mxffts = {
            (True, True): tk.fft.irfftn,
            (True, False): tk.fft.rfftn,
            (False, True): tk.fft.ifftn,
            (False, False): tk.fft.fftn,
        }
        shape = (3, 8, 6)
        r = np.random.rand(*shape).astype(np.float32)
        i = np.random.rand(*shape).astype(np.float32)
        for (real, inverse), fftn in mxffts.items():
            a_np = r if real and not inverse else r + 1j * i
            for axes in [(-1,), (0,), (-2, -1), (-1, -2), (0, 1)]:
                yield fftn, a_np, axes

    def test_fft_vmap(self):
        for fftn, a_np, axes in self.make_ffts():
            a = tk.array(a_np)
            f = lambda x: fftn(x, axes=axes)
            expected = tk.stack([f(a[i]) for i in range(a.shape[0])])
            out = tk.vmap(f)(a)
            self.assertEqual(tuple(out.shape), tuple(expected.shape))
            np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)

    def test_fft_jvp(self):
        # The fft is linear so the jvp is the fft of the tangent
        for fftn, a_np, axes in self.make_ffts():
            a = tk.array(a_np)
            t = tk.array(np.random.rand(*a_np.shape).astype(a_np.dtype))
            f = lambda x: fftn(x, axes=axes)
            expected = f(t)
            out = tk.jvp(f, [a], [t])[1][0]
            self.assertEqual(tuple(out.shape), tuple(expected.shape))
            np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
