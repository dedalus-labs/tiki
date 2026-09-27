# Copyright © 2026 Dedalus Labs, Inc.

import unittest

import tiki as tk
import tiki_tests


def opaque_double():
    """A forward using a primitive that has no JVP, on whichever backend runs.

    On CUDA that is a custom kernel; on a CPU-enabled build it is SVD, whose
    primitive defines no jvp. Returns None when neither is available.
    """
    if tk.cuda.is_available():
        kernel = tk.fast.cuda_kernel(
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
        tk.eval(tk.linalg.svd(tk.eye(2), stream=tk.cpu))
    except Exception:
        return None

    def forward(x):
        u, s, vt = tk.linalg.svd(tk.diag(x), stream=tk.cpu)
        return 2 * tk.diag(u @ tk.diag(s) @ vt)

    return forward


class TestJvpStopGradient(tiki_tests.TIKITestCase):
    def test_completed_jvp_outputs_evaluate_in_independent_transforms(self):
        # Completed JVP outputs have no stale tracers, even on stopped paths.
        # Witness: async_eval of a stopped square during an independent grad.
        (out,), (tangent,) = tk.jvp(
            lambda x: tk.stop_gradient(x * x), [tk.array(2.0)], [tk.array(1.0)]
        )
        grad = tk.grad(lambda z: (tk.async_eval(out), z * z)[1])(tk.array(3.0))
        self.assertEqual(grad.item(), 6.0)
        self.assertEqual(out.item(), 4.0)
        self.assertEqual(tangent.item(), 0.0)

    # Invariant: forward mode does not tape anything upstream of stop_gradient,
    # so a custom_function whose forward uses a primitive without a JVP still
    # differentiates through its registered rule.
    # Witness: doubling through an opaque forward, jvp rule 2 * tangent.
    def test_custom_jvp_rule_over_opaque_forward(self):
        forward = opaque_double()
        if forward is None:
            self.skipTest("no backend with a JVP-less primitive available")
        double = tk.custom_function(forward)
        # For a single-input function Tiki passes the primal and tangent as bare arrays.
        double.jvp(lambda primal, tangent: 2 * tangent)
        x = tk.arange(4, dtype=tk.float32)
        t = tk.array([1.0, 0.0, 3.0, 0.0])
        out, tangent = tk.jvp(double, (x,), (t,))
        tk.eval(out, tangent)
        self.assertTrue(tk.array_equal(out[0], 2 * x))
        self.assertTrue(tk.array_equal(tangent[0], 2 * t))

    # Invariant: a subgraph reachable only through stop_gradient is not taped,
    # while the same subgraph reachable through a live path still is.
    # Witness: out = stop_gradient(g(x)) + 3 x has tangent 3 t; out2 uses g live.
    def test_stop_gradient_prunes_only_dead_paths(self):
        def g(x):
            return x * x

        def dead(x):
            return tk.stop_gradient(g(x)) + 3 * x

        def live(x):
            y = g(x)
            return tk.stop_gradient(y) + y

        x = tk.array([1.0, 2.0])
        t = tk.array([1.0, 1.0])
        self.assertTrue(
            tk.array_equal(tk.jvp(dead, (x,), (t,))[1][0], tk.array([3.0, 3.0]))
        )
        self.assertTrue(tk.array_equal(tk.jvp(live, (x,), (t,))[1][0], 2 * x))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
