# Copyright © 2023 Apple Inc.

import math
import unittest

import tiki as tk
import tiki_tests


class TestRandom(tiki_tests.TIKITestCase):
    def test_global_rng(self):
        tk.random.seed(3)
        a = tk.random.uniform()
        b = tk.random.uniform()

        tk.random.seed(3)
        x = tk.random.uniform()
        y = tk.random.uniform()

        self.assertEqual(a.item(), x.item())
        self.assertEqual(y.item(), b.item())

    def test_key(self):
        k1 = tk.random.key(0)
        k2 = tk.random.key(0)
        self.assertTrue(tk.array_equal(k1, k2))

        k2 = tk.random.key(1)
        self.assertFalse(tk.array_equal(k1, k2))

    def test_key_split(self):
        key = tk.random.key(0)

        k1, k2 = tk.random.split(key)
        self.assertFalse(tk.array_equal(k1, k2))

        r1, r2 = tk.random.split(key)
        self.assertTrue(tk.array_equal(k1, r1))
        self.assertTrue(tk.array_equal(k2, r2))

        keys = tk.random.split(key, 10)
        self.assertEqual(keys.shape, (10, 2))

    def test_uniform(self):
        key = tk.random.key(0)
        a = tk.random.uniform(key=key)
        self.assertEqual(a.shape, ())
        self.assertEqual(a.dtype, tk.float32)

        b = tk.random.uniform(key=key)
        self.assertEqual(a.item(), b.item())

        a = tk.random.uniform(shape=(2, 3))
        self.assertEqual(a.shape, (2, 3))

        a = tk.random.uniform(shape=(1000,), low=-1, high=5)
        self.assertTrue(tk.all((a > -1) < 5).item())

        a = tk.random.uniform(shape=(1000,), low=tk.array(-1), high=5)
        self.assertTrue(tk.all((a > -1) < 5).item())

        a = tk.random.uniform(low=-0.1, high=0.1, shape=(1,), dtype=tk.bfloat16)
        self.assertEqual(a.dtype, tk.bfloat16)

        self.assertEqual(tk.random.uniform().dtype, tk.random.uniform(dtype=None).dtype)

        with self.assertRaises(ValueError):
            tk.random.uniform(shape=(2, -3))

    def test_normal_and_laplace(self):
        # Same tests for normal and laplace.
        for distribution_sampler in [tk.random.normal, tk.random.laplace]:
            key = tk.random.key(0)
            a = distribution_sampler(key=key)
            self.assertEqual(a.shape, ())
            self.assertEqual(a.dtype, tk.float32)

            b = distribution_sampler(key=key)
            self.assertEqual(a.item(), b.item())

            a = distribution_sampler(shape=(2, 3))
            self.assertEqual(a.shape, (2, 3))

            ## Generate in float16 or bfloat16
            for t in [tk.float16, tk.bfloat16]:
                a = distribution_sampler(dtype=t)
                self.assertEqual(a.dtype, t)

            # Generate with a given mean and standard deviation
            loc = 1.0
            scale = 2.0

            a = distribution_sampler(shape=(3, 2), loc=loc, scale=scale, key=key)
            b = scale * distribution_sampler(shape=(3, 2), key=key) + loc
            self.assertTrue(tk.allclose(a, b))

            a = distribution_sampler(
                shape=(3, 2), loc=loc, scale=scale, dtype=tk.float16, key=key
            )
            b = (
                scale * distribution_sampler(shape=(3, 2), dtype=tk.float16, key=key)
                + loc
            )
            self.assertTrue(tk.allclose(a, b))

            self.assertEqual(
                distribution_sampler().dtype, distribution_sampler(dtype=None).dtype
            )

            # Test not getting -inf or inf with half precison
            for hp in [tk.float16, tk.bfloat16]:
                a = abs(distribution_sampler(shape=(10000,), loc=0, scale=1, dtype=hp))
                self.assertTrue(tk.all(a < tk.inf))

    def test_multivariate_normal(self):
        key = tk.random.key(0)
        mean = tk.array([0, 0])
        cov = tk.array([[1, 0], [0, 1]])

        a = tk.random.multivariate_normal(mean, cov, key=key, stream=tk.cpu)
        self.assertEqual(a.shape, (2,))

        ## Check dtypes
        for t in [tk.float32]:
            a = tk.random.multivariate_normal(
                mean, cov, dtype=t, key=key, stream=tk.cpu
            )
            self.assertEqual(a.dtype, t)
        for t in [
            tk.int8,
            tk.int32,
            tk.int64,
            tk.uint8,
            tk.uint32,
            tk.uint64,
            tk.float16,
            tk.bfloat16,
        ]:
            with self.assertRaises(ValueError):
                tk.random.multivariate_normal(
                    mean, cov, dtype=t, key=key, stream=tk.cpu
                )

        ## Check incompatible shapes
        with self.assertRaises(ValueError):
            mean = tk.zeros((2, 2))
            cov = tk.zeros((2, 2))
            tk.random.multivariate_normal(mean, cov, shape=(3,), key=key, stream=tk.cpu)

        with self.assertRaises(ValueError):
            mean = tk.zeros((2))
            cov = tk.zeros((2, 2, 2))
            tk.random.multivariate_normal(mean, cov, shape=(3,), key=key, stream=tk.cpu)

        with self.assertRaises(ValueError):
            mean = tk.zeros((3,))
            cov = tk.zeros((2, 2))
            tk.random.multivariate_normal(mean, cov, key=key, stream=tk.cpu)

        with self.assertRaises(ValueError):
            mean = tk.zeros((2,))
            cov = tk.zeros((2, 3))
            tk.random.multivariate_normal(mean, cov, key=key, stream=tk.cpu)

        ## Different shape of mean and cov
        mean = tk.array([[0, 7], [1, 2], [3, 4]])
        cov = tk.array([[1, 0.5], [0.5, 1]])
        a = tk.random.multivariate_normal(mean, cov, shape=(4, 3), stream=tk.cpu)
        self.assertEqual(a.shape, (4, 3, 2))

        ## Check correcteness of the mean and covariance
        n_test = int(1e5)

        def check_jointly_gaussian(data, mean, cov):
            empirical_mean = tk.mean(data, axis=0)
            empirical_cov = (
                (data - empirical_mean).T @ (data - empirical_mean) / data.shape[0]
            )
            N = data.shape[1]
            self.assertTrue(
                tk.allclose(
                    empirical_mean, mean, rtol=0.0, atol=10 * N**2 / math.sqrt(n_test)
                )
            )
            self.assertTrue(
                tk.allclose(
                    empirical_cov, cov, rtol=0.0, atol=10 * N**2 / math.sqrt(n_test)
                )
            )

        mean = tk.array([4.0, 7.0])
        cov = tk.array([[2, 0.5], [0.5, 1]])
        data = tk.random.multivariate_normal(
            mean, cov, shape=(n_test,), key=key, stream=tk.cpu
        )
        check_jointly_gaussian(data, mean, cov)

        mean = tk.arange(3)
        cov = tk.array([[1, -1, 0.5], [-1, 1, -0.5], [0.5, -0.5, 1]])
        data = tk.random.multivariate_normal(
            mean, cov, shape=(n_test,), key=key, stream=tk.cpu
        )
        check_jointly_gaussian(data, mean, cov)

    def test_randint(self):
        a = tk.random.randint(0, 1, [])
        self.assertEqual(a.shape, ())
        self.assertEqual(a.dtype, tk.int32)

        shape = (88,)
        low = tk.array(3)
        high = tk.array(15)

        key = tk.random.key(0)
        a = tk.random.randint(low, high, shape, key=key)
        self.assertEqual(a.shape, shape)
        self.assertEqual(a.dtype, tk.int32)

        # Check using the same key yields the same value
        b = tk.random.randint(low, high, shape, key=key)
        self.assertListEqual(a.tolist(), b.tolist())

        shape = (3, 4)
        low = tk.reshape(tk.array([0] * 3), [3, 1])
        high = tk.reshape(tk.array([12, 13, 14, 15]), [1, 4])

        a = tk.random.randint(low, high, shape)
        self.assertEqual(a.shape, shape)

        a = tk.random.randint(-10, 10, [1000, 1000])
        self.assertTrue(tk.all(-10 <= a).item() and tk.all(a < 10).item())

        a = tk.random.randint(10, -10, [1000, 1000])
        self.assertTrue(tk.all(a == 10).item())

        # Bounds hold when the interval is not exactly representable in float32
        for dtype, low, high in [
            (tk.int32, 2**24, 2**24 + 2),
            (tk.uint32, 2**24, 2**24 + 2),
            (tk.int64, 2**40, 2**40 + 1024),
        ]:
            a = tk.random.randint(low, high, [10000], dtype=dtype, key=key)
            self.assertTrue(tk.all(a >= low).item())
            self.assertTrue(tk.all(a < high).item())

        # The lower bound is reachable when the interval spans negative values
        a = tk.random.randint(-5, 5, [20000], key=key)
        self.assertEqual(sorted(set(a.tolist())), list(range(-5, 5)))

        # Booleans use the whole interval
        a = tk.random.randint(0, 2, [1000], dtype=tk.bool_, key=key)
        self.assertEqual(sorted(set(a.tolist())), [False, True])
        a = tk.random.randint(0, 1, [1000], dtype=tk.bool_, key=key)
        self.assertFalse(tk.any(a).item())

        self.assertEqual(
            tk.random.randint(0, 1).dtype, tk.random.randint(0, 1, dtype=None).dtype
        )

    def test_bernoulli(self):
        a = tk.random.bernoulli()
        self.assertEqual(a.shape, ())
        self.assertEqual(a.dtype, tk.bool_)

        a = tk.random.bernoulli(tk.array(0.5), [5])
        self.assertEqual(a.shape, (5,))

        a = tk.random.bernoulli(tk.array([2.0, -2.0]))
        self.assertEqual(a.tolist(), [True, False])
        self.assertEqual(a.shape, (2,))

        p = tk.array([0.1, 0.2, 0.3])
        tk.reshape(p, [1, 3])
        x = tk.random.bernoulli(p, [4, 3])
        self.assertEqual(x.shape, (4, 3))

        with self.assertRaises(ValueError):
            tk.random.bernoulli(p, [2])  # Bad shape

        with self.assertRaises(ValueError):
            tk.random.bernoulli(0, [2])  # Bad type

    def test_truncated_normal(self):
        a = tk.random.truncated_normal(-2.0, 2.0)
        self.assertEqual(a.size, 1)
        self.assertEqual(a.dtype, tk.float32)

        a = tk.random.truncated_normal(tk.array([]), tk.array([]))
        self.assertEqual(a.dtype, tk.float32)
        self.assertEqual(a.size, 0)

        lower = tk.reshape(tk.array([-2.0, 0.0]), [1, 2])
        upper = tk.reshape(tk.array([0.0, 1.0, 2.0]), [3, 1])
        a = tk.random.truncated_normal(lower, upper)

        self.assertEqual(a.shape, (3, 2))
        self.assertTrue(tk.all(lower <= a).item() and tk.all(a <= upper).item())

        a = tk.random.truncated_normal(2.0, -2.0)
        self.assertTrue(tk.all(a == 2.0).item())

        a = tk.random.truncated_normal(-3.0, 3.0, [542, 399])
        self.assertEqual(a.shape, (542, 399))

        lower = tk.array([-2.0, -1.0])
        higher = tk.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            tk.random.truncated_normal(lower, higher)  # Bad shape

        self.assertEqual(
            tk.random.truncated_normal(0, 1).dtype,
            tk.random.truncated_normal(0, 1, dtype=None).dtype,
        )

    def test_gumbel(self):
        samples = tk.random.gumbel(shape=(100, 100))
        self.assertEqual(samples.shape, (100, 100))
        self.assertEqual(samples.dtype, tk.float32)
        mean = 0.5772
        # Std deviation of the sample mean is small (<0.02),
        # so this test is pretty conservative
        self.assertTrue(tk.abs(tk.mean(samples) - mean) < 0.2)

        self.assertEqual(
            tk.random.gumbel((1, 1)).dtype, tk.random.gumbel((1, 1), dtype=None).dtype
        )

    def test_categorical(self):
        logits = tk.zeros((10, 20))
        self.assertEqual(tk.random.categorical(logits, -1).shape, (10,))
        self.assertEqual(tk.random.categorical(logits, 0).shape, (20,))
        self.assertEqual(tk.random.categorical(logits, 1).shape, (10,))

        out = tk.random.categorical(logits)
        self.assertEqual(out.shape, (10,))
        self.assertEqual(out.dtype, tk.uint32)
        self.assertTrue(tk.max(out).item() < 20)

        out = tk.random.categorical(logits, 0, [5, 20])
        self.assertEqual(out.shape, (5, 20))
        self.assertTrue(tk.max(out).item() < 10)

        out = tk.random.categorical(logits, 1, num_samples=7)
        self.assertEqual(out.shape, (10, 7))
        out = tk.random.categorical(logits, 0, num_samples=7)
        self.assertEqual(out.shape, (20, 7))

        with self.assertRaises(ValueError):
            tk.random.categorical(logits, shape=[10, 5], num_samples=5)

        # Single distribution.
        logits = tk.zeros((20,))

        out = tk.random.categorical(logits, num_samples=7)
        self.assertEqual(out.shape, (7,))
        self.assertEqual(out.dtype, tk.uint32)
        self.assertTrue(tk.max(out).item() < 20)

        out = tk.random.categorical(logits, 0, [5, 3])
        self.assertEqual(out.shape, (5, 3))
        self.assertTrue(tk.max(out).item() < 20)

        self.assertEqual(tk.random.categorical(logits, num_samples=1).shape, (1,))
        self.assertEqual(tk.random.categorical(logits, num_samples=0).shape, (0,))

    def test_permutation(self):
        x = sorted(tk.random.permutation(4).tolist())
        self.assertEqual([0, 1, 2, 3], x)

        x = tk.array([0, 1, 2, 3])
        x = sorted(tk.random.permutation(x).tolist())
        self.assertEqual([0, 1, 2, 3], x)

        x = tk.array([0, 1, 2, 3])
        x = sorted(tk.random.permutation(x).tolist())

        # 2-D
        x = tk.arange(16).reshape(4, 4)
        out = tk.sort(tk.random.permutation(x, axis=0), axis=0)
        self.assertTrue(tk.array_equal(x, out))
        out = tk.sort(tk.random.permutation(x, axis=1), axis=1)
        self.assertTrue(tk.array_equal(x, out))

        # Basically 0 probability this should fail.
        sorted_x = tk.arange(16384)
        x = tk.random.permutation(16384)
        self.assertFalse(tk.array_equal(sorted_x, x))

        # Preserves shape / doesn't cast input to int
        x = tk.random.permutation(tk.array([[1]]))
        self.assertEqual(x.shape, (1, 1))

    def test_complex_normal(self):
        sample = tk.random.normal(tuple(), dtype=tk.complex64)
        self.assertEqual(sample.shape, tuple())
        self.assertEqual(sample.dtype, tk.complex64)

        sample = tk.random.normal((1, 2, 3, 4), dtype=tk.complex64)
        self.assertEqual(sample.shape, (1, 2, 3, 4))
        self.assertEqual(sample.dtype, tk.complex64)

        sample = tk.random.normal((1, 2, 3, 4), dtype=tk.complex64, scale=2.0, loc=3.0)
        self.assertEqual(sample.shape, (1, 2, 3, 4))
        self.assertEqual(sample.dtype, tk.complex64)

        sample = tk.random.normal(
            (1, 2, 3, 4), dtype=tk.complex64, scale=2.0, loc=3.0 + 1j
        )
        self.assertEqual(sample.shape, (1, 2, 3, 4))
        self.assertEqual(sample.dtype, tk.complex64)

    def test_broadcastable_scale_loc(self):
        b = tk.random.normal((10, 2))
        sample = tk.random.normal((2, 10, 2), loc=b, scale=b)
        tk.eval(sample)
        self.assertEqual(sample.shape, (2, 10, 2))

        with self.assertRaises(ValueError):
            b = tk.random.normal((10,))
            sample = tk.random.normal((2, 10, 2), loc=b, scale=b)

        b = tk.random.normal((3, 1, 2))
        sample = tk.random.normal((3, 4, 2), dtype=tk.float16, loc=b, scale=b)
        tk.eval(sample)
        self.assertEqual(sample.shape, (3, 4, 2))
        self.assertEqual(sample.dtype, tk.float16)


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
