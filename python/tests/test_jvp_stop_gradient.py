# Copyright © 2026 Dedalus Labs, Inc.

import unittest

import mlx.core as mx
import mlx_tests


def opaque_double():
    """A forward using a primitive that has no JVP, on whichever backend runs.

    On CUDA that is a custom kernel; on a CPU-enabled build it is SVD, whose
    primitive defines no jvp. Returns None when neither is available.
    """
    if mx.cuda.is_available():
        kernel = mx.fast.cuda_kernel(
            name="double_it",
            input_names=["x"],
            output_names=["y"],
            source="int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < N) y[i] = 2 * x[i];",
        )

        def forward(x):
            return kernel(
                inputs=[x],
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
                template=[("N", x.size)],
                grid=(x.size, 1, 1),
                threadgroup=(min(x.size, 128), 1, 1),
            )[0]

        return forward
    try:
        mx.eval(mx.linalg.svd(mx.eye(2), stream=mx.cpu))
    except Exception:
        return None

    def forward(x):
        u, s, vt = mx.linalg.svd(mx.diag(x), stream=mx.cpu)
        return 2 * mx.diag(u @ mx.diag(s) @ vt)

    return forward


class TestJvpStopGradient(mlx_tests.MLXTestCase):
    # Invariant: forward mode does not tape anything upstream of stop_gradient,
    # so a custom_function whose forward uses a primitive without a JVP still
    # differentiates through its registered rule.
    # Witness: doubling through an opaque forward, jvp rule 2 * tangent.
    def test_custom_jvp_rule_over_opaque_forward(self):
        forward = opaque_double()
        if forward is None:
            self.skipTest("no backend with a JVP-less primitive available")
        double = mx.custom_function(forward)
        # For a single-input function MLX passes the primal and tangent as bare arrays.
        double.jvp(lambda primal, tangent: 2 * tangent)
        x = mx.arange(4, dtype=mx.float32)
        t = mx.array([1.0, 0.0, 3.0, 0.0])
        out, tangent = mx.jvp(double, (x,), (t,))
        mx.eval(out, tangent)
        self.assertTrue(mx.array_equal(out[0], 2 * x))
        self.assertTrue(mx.array_equal(tangent[0], 2 * t))

    # Invariant: a subgraph reachable only through stop_gradient is not taped,
    # while the same subgraph reachable through a live path still is.
    # Witness: out = stop_gradient(g(x)) + 3 x has tangent 3 t; out2 uses g live.
    def test_stop_gradient_prunes_only_dead_paths(self):
        def g(x):
            return x * x

        def dead(x):
            return mx.stop_gradient(g(x)) + 3 * x

        def live(x):
            y = g(x)
            return mx.stop_gradient(y) + y

        x = mx.array([1.0, 2.0])
        t = mx.array([1.0, 1.0])
        self.assertTrue(
            mx.array_equal(mx.jvp(dead, (x,), (t,))[1][0], mx.array([3.0, 3.0]))
        )
        self.assertTrue(mx.array_equal(mx.jvp(live, (x,), (t,))[1][0], 2 * x))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
