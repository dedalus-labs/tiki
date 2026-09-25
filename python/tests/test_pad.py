import mlx.core as mx
import mlx_tests


class TestPad(mlx_tests.MLXTestCase):
    def test_source_vjp_preserves_batched_axes(self):
        value = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4)
        for axis in (0, 1, 2):
            padded = mx.vmap(
                lambda x: mx.pad(x, ((1, 2), (2, 1)), constant_values=3),
                in_axes=axis,
                out_axes=axis,
            )
            gradient = mx.grad(
                lambda x, transform=padded: mx.sum(mx.square(transform(x)))
            )(value)
            self.assertEqualArray(gradient, 2 * value, rtol=0, atol=0)

    def test_source_jvp_excludes_constant_padding(self):
        value = mx.arange(6, dtype=mx.float32).reshape(2, 3)
        tangent = mx.ones_like(value)

        def function(x):
            return mx.pad(x, ((1, 2), (2, 1)), constant_values=3)

        _, derivatives = mx.jvp(function, [value], [tangent])
        expected = function(tangent) - function(mx.zeros_like(tangent))
        self.assertEqualArray(derivatives[0], expected, rtol=0, atol=0)

    def test_padding_value_derivatives_are_explicitly_unsupported(self):
        value = mx.ones((2, 3))

        def function(fill):
            return mx.pad(value, ((1, 2), (2, 1)), constant_values=fill)

        with self.assertRaisesRegex(ValueError, "padding value"):
            mx.grad(lambda fill: mx.sum(function(fill)))(mx.array(3.0))
        with self.assertRaisesRegex(ValueError, "padding value"):
            mx.jvp(function, [mx.array(3.0)], [mx.array(1.0)])


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
