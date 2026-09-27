import tiki as tk
import tiki_tests


class TestPad(tiki_tests.TIKITestCase):
    def test_source_vjp_preserves_batched_axes(self):
        value = tk.arange(24, dtype=tk.float32).reshape(2, 3, 4)
        for axis in (0, 1, 2):
            padded = tk.vmap(
                lambda x: tk.pad(x, ((1, 2), (2, 1)), constant_values=3),
                in_axes=axis,
                out_axes=axis,
            )
            gradient = tk.grad(
                lambda x, transform=padded: tk.sum(tk.square(transform(x)))
            )(value)
            self.assertEqualArray(gradient, 2 * value, rtol=0, atol=0)

    def test_source_jvp_excludes_constant_padding(self):
        value = tk.arange(6, dtype=tk.float32).reshape(2, 3)
        tangent = tk.ones_like(value)

        def function(x):
            return tk.pad(x, ((1, 2), (2, 1)), constant_values=3)

        _, derivatives = tk.jvp(function, [value], [tangent])
        expected = function(tangent) - function(tk.zeros_like(tangent))
        self.assertEqualArray(derivatives[0], expected, rtol=0, atol=0)

    def test_padding_value_derivatives_are_explicitly_unsupported(self):
        value = tk.ones((2, 3))

        def function(fill):
            return tk.pad(value, ((1, 2), (2, 1)), constant_values=fill)

        with self.assertRaisesRegex(ValueError, "padding value"):
            tk.grad(lambda fill: tk.sum(function(fill)))(tk.array(3.0))
        with self.assertRaisesRegex(ValueError, "padding value"):
            tk.jvp(function, [tk.array(3.0)], [tk.array(1.0)])


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
