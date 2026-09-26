"""Every output and cotangent belongs to the captured forward program."""

import unittest

import mlx.core as mx

import tiki as tk
from associative_scan import ScanOp
from graph import capture, replay
from scan_lowering import ScanSchedule


def scalar_operations(left: mx.array, right: mx.array) -> tuple[mx.array, ...]:
    constant = mx.broadcast_to(mx.array(2.0), left.shape)
    return (
        left + right,
        left - right,
        left * right,
        -left,
        mx.square(left),
        mx.rsqrt(left),
        constant,
    )


class ReplayTests(unittest.TestCase):
    def test_all_scalar_operations_preserve_output_order(self) -> None:
        inputs = (mx.array([1.0, 4.0, 9.0]), mx.array([2.0, 3.0, 5.0]))
        graph = capture(scalar_operations, (((3,), (1,)),) * 2)
        outputs = replay(graph, inputs)
        self.assertEqual(len(outputs), 7)
        for actual, expected in zip(outputs, scalar_operations(*inputs)):
            self.assertTrue(mx.array_equal(actual, expected))

    def test_context_operations_keep_their_output_shapes(self) -> None:
        def function(array: mx.array) -> tuple[mx.array, ...]:
            return mx.sum(array, axis=-1, keepdims=True), array.T

        array = mx.arange(6, dtype=mx.float32).reshape(2, 3)
        graph = capture(function, (((2, 3), (3, 1)),))
        outputs = replay(graph, (array,))
        self.assertEqual(tuple(output.shape for output in outputs), ((2, 1), (3, 2)))
        for actual, expected in zip(outputs, function(array)):
            self.assertTrue(mx.array_equal(actual, expected))


@unittest.skipUnless(mx.cuda.is_available(), "requires MLX CUDA")
class CompiledReplayTests(unittest.TestCase):
    def test_scan_derivatives_and_recursive_tiles_share_the_frozen_combine(
        self,
    ) -> None:
        coefficient = [0.5]

        def combine(left: mx.array, right: mx.array) -> tuple[mx.array, ...]:
            return (coefficient[0] * left * right,)

        scan = ScanOp(combine, 1, 0, ScanSchedule(threads=32, elements_per_thread=1))
        coefficient[0] = 0.25
        for length in (3, 65, 1025):
            with self.subTest(length=length):
                array = mx.full((length,), 2.0)
                outputs, gradients = mx.vjp(scan, (array,), (mx.ones_like(array),))
                _, tangents = mx.jvp(scan, (array,), (mx.ones_like(array),))
                self.assertTrue(mx.array_equal(outputs[0], array))
                self.assertTrue(mx.array_equal(gradients[0], mx.arange(length, 0, -1)))
                self.assertTrue(mx.array_equal(tangents[0], mx.arange(1, length + 1)))

    def test_scalar_vocabulary_lowers_with_multiple_outputs(self) -> None:
        compiled = tk.compile()(scalar_operations)
        inputs = (mx.array([1.0, 4.0, 9.0]), mx.array([2.0, 3.0, 5.0]))
        for actual, expected in zip(compiled(*inputs), scalar_operations(*inputs)):
            self.assertTrue(mx.allclose(actual, expected, atol=1e-6))

    def test_all_cotangents_use_the_frozen_forward_after_eviction(self) -> None:
        coefficient = [2.0]

        def function(array: mx.array) -> tuple[mx.array, ...]:
            return coefficient[0] * array, array * array

        compiled = tk.compile()(function)

        def evict(array: mx.array) -> tuple[mx.array, ...]:
            outputs = compiled(array)
            tk.specialize.cache_clear()
            tk.differentiable.cache_clear()
            coefficient[0] = 9.0
            return outputs

        array = mx.array([1.0, 2.0, 3.0])
        cotangents = (mx.full((3,), 3.0), mx.full((3,), 5.0))
        outputs, gradients = mx.vjp(evict, (array,), cotangents)
        self.assertTrue(mx.array_equal(outputs[0], 2 * array))
        self.assertTrue(mx.array_equal(gradients[0], 6 + 10 * array))

    def test_jvp_returns_every_output_tangent(self) -> None:
        compiled = tk.compile()(lambda array: (2 * array, array * array))
        array, tangent = mx.array([1.0, 2.0, 3.0]), mx.full((3,), 4.0)
        _, tangents = mx.jvp(compiled, (array,), (tangent,))
        self.assertEqual(len(tangents), 2)
        self.assertTrue(mx.array_equal(tangents[0], 2 * tangent))
        self.assertTrue(mx.array_equal(tangents[1], 2 * array * tangent))

    def test_second_derivatives_remain_registered_for_all_outputs(self) -> None:
        compiled = tk.compile()(lambda array: (array, array * array))

        def loss(array: mx.array) -> mx.array:
            linear, squared = compiled(array)
            return mx.sum(3 * linear + 5 * squared)

        gradient = mx.grad(loss)
        array = mx.array([1.0, 2.0, 3.0])
        second = mx.grad(lambda value: mx.sum(gradient(value)))(array)
        self.assertTrue(mx.array_equal(second, mx.full((3,), 10.0)))


if __name__ == "__main__":
    unittest.main()
