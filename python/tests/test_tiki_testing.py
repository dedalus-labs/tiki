# Copyright © 2026 Dedalus Labs, Inc.

"""``check_grads`` passes on derivatives MLX provides and catches each bug class
it is designed for: a wrong sign, swapped operands, a stride-dependent backward,
a nondeterministic backward, a wrong vmap rule, and a wrong second derivative."""

import itertools
import unittest

import mlx.core as mx
from mlx.tiki.testing import GradcheckError, check_grads


def normal(shape: tuple[int, ...], seed: int) -> mx.array:
    return mx.random.normal(shape, key=mx.random.key(seed))


class PassingFunctions(unittest.TestCase):
    def test_pytree_function_passes_every_mode_at_second_order(self) -> None:
        def f(inputs: dict[str, mx.array], y: mx.array) -> dict[str, mx.array]:
            x = inputs["x"]
            return {"s": mx.sin(x) @ y, "t": (x * inputs["w"]).sum()}

        args = ({"x": normal((3, 4), 1), "w": normal((3, 4), 2)}, normal((4, 2), 3))
        check_grads(f, args, order=2, full_jacobian=True)

    def test_float64_on_the_cpu_stream(self) -> None:
        with mx.stream(mx.cpu):
            x = normal((6,), 4).astype(mx.float64)
            check_grads(
                lambda x: mx.exp(x) * mx.cos(x), (x,), order=2, full_jacobian=True
            )

    def test_reference_replaces_finite_differences_for_a_float32_only_kernel(
        self,
    ) -> None:
        @mx.custom_function
        def triple(x: mx.array) -> mx.array:
            if x.dtype != mx.float32:
                raise TypeError("the kernel runs float32 only")
            return x * 3

        @triple.jvp
        def _(primals: mx.array, tangents: mx.array) -> mx.array:
            return tangents * 3

        @triple.vjp
        def _(primals: mx.array, cotangent: mx.array, output: mx.array) -> mx.array:
            return cotangent * 3

        x = normal((4, 5), 5)
        with self.assertRaisesRegex(GradcheckError, "reference="):
            check_grads(triple, (x,))
        check_grads(triple, (x,), reference=lambda x: x * 3)

    def test_argument_validation(self) -> None:
        x = normal((3,), 6)
        with self.assertRaises(TypeError):
            check_grads(mx.sin, x)
        with self.assertRaises(ValueError):
            check_grads(mx.sin, (x,), modes=("fwd", "hvp"))
        with self.assertRaises(TypeError):
            check_grads(mx.sin, (mx.arange(3),))
        with self.assertRaises(ValueError):
            check_grads(mx.sin, (x,), order=0)


class PlantedBugs(unittest.TestCase):
    def test_wrong_vjp_sign_breaks_the_transpose_identity(self) -> None:
        @mx.custom_function
        def f(x: mx.array) -> mx.array:
            return mx.exp(x)

        @f.vjp
        def _(primals: mx.array, cotangent: mx.array, output: mx.array) -> mx.array:
            return -cotangent * output

        with self.assertRaisesRegex(GradcheckError, "rev: transpose identity"):
            check_grads(f, (normal((5,), 7),))

    def test_swapped_jvp_operands_disagree_with_finite_differences(self) -> None:
        @mx.custom_function
        def f(x: mx.array, y: mx.array) -> mx.array:
            return x * y

        @f.jvp
        def _(
            primals: tuple[mx.array, mx.array], tangents: tuple[mx.array, mx.array]
        ) -> mx.array:
            x, y = primals
            dx, dy = tangents
            return dx * x + dy * y

        with self.assertRaisesRegex(GradcheckError, "dense: fwd"):
            check_grads(f, (normal((5,), 8), normal((5,), 9)), modes=("fwd",))

    def test_consistently_wrong_derivatives_need_the_reference(self) -> None:
        @mx.custom_function
        def f(x: mx.array) -> mx.array:
            if x.dtype != mx.float32:
                raise TypeError("float32 only")
            return x * 3

        @f.jvp
        def _(primals: mx.array, tangents: mx.array) -> mx.array:
            return tangents * 6

        @f.vjp
        def _(primals: mx.array, cotangent: mx.array, output: mx.array) -> mx.array:
            return cotangent * 6

        x = normal((5,), 10)
        check_grads(f, (x,), modes=("rev",), reference=None) if False else None
        with self.assertRaisesRegex(GradcheckError, "fwd"):
            check_grads(f, (x,), reference=lambda x: x * 3)

    @unittest.skipUnless(hasattr(mx.array, "strides"), "needs the Tiki strides binding")
    def test_stride_dependent_backward_fails_under_a_view(self) -> None:
        @mx.custom_function
        def f(x: mx.array) -> mx.array:
            return x * 2

        @f.vjp
        def _(primals: mx.array, cotangent: mx.array, output: mx.array) -> mx.array:
            row_major = primals.strides == (primals.shape[1], 1)
            return cotangent * 2 if row_major else cotangent

        x = normal((3, 4), 11)
        check_grads(f, (x,), layouts=False)
        with self.assertRaisesRegex(GradcheckError, "transposed"):
            check_grads(f, (x,))

    def test_nondeterministic_backward(self) -> None:
        calls = itertools.count()

        @mx.custom_function
        def f(x: mx.array) -> mx.array:
            return x * 2

        @f.vjp
        def _(primals: mx.array, cotangent: mx.array, output: mx.array) -> mx.array:
            return cotangent * (2 + 1e-4 * next(calls))

        x = normal((5,), 12)
        with self.assertRaisesRegex(GradcheckError, "nondeterministic"):
            check_grads(f, (x,), modes=("rev",))
        check_grads(f, (x,), modes=("rev",), nondeterminism=1e-2)

    def test_wrong_vmap_rule(self) -> None:
        @mx.custom_function
        def f(x: mx.array) -> mx.array:
            return x * 3

        @f.vmap
        def _(inputs: mx.array, axes: int) -> tuple[mx.array, int]:
            return inputs * 2, axes

        with self.assertRaisesRegex(GradcheckError, "vmap forward"):
            check_grads(f, (normal((4,), 13),), modes=("vmap",))

    def test_wrong_second_derivative_is_caught_only_at_order_two(self) -> None:
        @mx.custom_function
        def cube_backward(x: mx.array, cotangent: mx.array) -> mx.array:
            return 3 * x * x * cotangent

        @cube_backward.vjp
        def _(
            primals: tuple[mx.array, mx.array], cotangent: mx.array, output: mx.array
        ) -> tuple[mx.array, mx.array]:
            x, first = primals
            return 3 * x * first * cotangent, 3 * x * x * cotangent

        @mx.custom_function
        def cube(x: mx.array) -> mx.array:
            return x**3

        @cube.vjp
        def _(primals: mx.array, cotangent: mx.array, output: mx.array) -> mx.array:
            return cube_backward(primals, cotangent)

        x = normal((5,), 14)
        check_grads(cube, (x,), order=1)
        with self.assertRaisesRegex(GradcheckError, "> vjp: rev"):
            check_grads(cube, (x,), order=2)


if __name__ == "__main__":
    unittest.main()
