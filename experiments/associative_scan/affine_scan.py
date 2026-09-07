"""Fast affine scan on CUDA: Blelloch tree kernels with a registered backward.

This is Tiki's second tier, the analogue of JAX's native GPU ``cumsum``: an
authored kernel for one combine, the affine pair
``(a_l, b_l) * (a_r, b_r) = (a_r * a_l, a_r * b_l + b_r)``, registered on
``mx.custom_function`` with an explicit VJP. Its correctness oracle is the
first tier, ``associative_scan(affine, ...)``, and MLX's own differentiation
of that tree, exactly how JAX derives the gradient of its kernel from the
generic tree.

Contract: float32 ``[batch, time]`` inputs, ``batch > 0``,
``1 <= time <= 2048``. Longer rows need a hierarchical schedule (tile scans, a
scan of tile summaries, a carry pass) and raise ``ScanContractError``; there
is no fallback to the tree.
"""

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

Pair = tuple[mx.array, mx.array]

MAX_TIME = 2048
THREADS = 128
HEADER = Path(__file__).with_name("blelloch.cuh").read_text()

FORWARD_SOURCE = """
__shared__ float s[2][N];
size_t row = size_t(blockIdx.x) * T;
for (int j = threadIdx.x; j < N; j += blockDim.x) {
  s[0][j] = j < T ? a[row + j] : 1.f;
  s[1][j] = j < T ? b[row + j] : 0.f;
}
__syncthreads();
tree_scan(s);
for (int j = threadIdx.x; j < T; j += blockDim.x) {
  p[row + j] = j ? a[row + j] * s[0][j] : a[row];
  h[row + j] = j ? a[row + j] * s[1][j] + b[row + j] : b[row];
}
"""

BACKWARD_SOURCE = """
__shared__ float s[3][N];
size_t row = size_t(blockIdx.x) * T;
for (int j = threadIdx.x; j < N; j += blockDim.x) {
  int t = T - 1 - j;
  s[0][j] = j < T ? (t + 1 < T ? a[row + t + 1] : 0.f) : 1.f;
  s[1][j] = j < T ? gp[row + t] : 0.f;
  s[2][j] = j < T ? gh[row + t] : 0.f;
}
__syncthreads();
tree_scan(s);
for (int j = threadIdx.x; j < T; j += blockDim.x) {
  int t = T - 1 - j;
  float coef = t + 1 < T ? a[row + t + 1] : 0.f;
  float rp = coef * s[1][j] + gp[row + t];
  float rh = coef * s[2][j] + gh[row + t];
  da[row + t] = t ? rp * p[row + t - 1] + rh * h[row + t - 1] : rp;
  db[row + t] = rh;
}
"""


class ScanContractError(ValueError):
    """The inputs are outside the kernel's contract."""


def _kernel(
    name: str, inputs: list[str], outputs: list[str], source: str
) -> Callable[..., list[mx.array]]:
    return mx.fast.cuda_kernel(
        name=name,
        input_names=inputs,
        output_names=outputs,
        header=HEADER,
        source=source,
    )


forward_kernel = _kernel("tiki_affine_scan_fwd", ["a", "b"], ["p", "h"], FORWARD_SOURCE)
backward_kernel = _kernel(
    "tiki_affine_scan_bwd", ["a", "p", "h", "gp", "gh"], ["da", "db"], BACKWARD_SOURCE
)


def _check_contract(inputs: tuple[mx.array, ...]) -> tuple[int, int]:
    if not inputs or any(not isinstance(array, mx.array) for array in inputs):
        raise ScanContractError("affine_scan inputs must be MLX arrays")
    shape = inputs[0].shape
    if len(shape) != 2:
        raise ScanContractError(
            f"affine_scan needs [batch, time] arrays, got shape {shape}"
        )
    for array in inputs:
        if array.shape != shape or array.dtype != mx.float32:
            raise ScanContractError(
                f"affine_scan needs matching float32 arrays, got {array.shape} {array.dtype}"
            )
    batch, time = shape
    if batch == 0 or not 1 <= time <= MAX_TIME:
        raise ScanContractError(
            f"affine_scan needs batch > 0 and 1 <= time <= {MAX_TIME}, got {shape}"
        )
    if batch > (2**31 - 1) // THREADS:
        raise ScanContractError(
            "affine_scan batch exceeds the signed 32-bit launch grid"
        )
    return batch, time


def _launch(kernel: Callable[..., list[mx.array]], *inputs: mx.array) -> Pair:
    batch, time = _check_contract(inputs)
    padded = 1 << (time - 1).bit_length()
    outputs = kernel(
        inputs=list(inputs),
        output_shapes=[inputs[0].shape, inputs[0].shape],
        output_dtypes=[mx.float32, mx.float32],
        template=[("T", time), ("N", padded)],
        grid=(batch * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        stream=mx.gpu,
    )
    return outputs[0], outputs[1]


def forward(a: mx.array, b: mx.array) -> Pair:
    """Prefix affine composition along time: returns (coefficient, offset) per step."""
    return _launch(forward_kernel, a, b)


def backward(primals: Pair, cotangents: Pair, outputs: Pair) -> Pair:
    a, _ = primals
    return _launch(backward_kernel, a, *outputs, *cotangents)


affine_scan = mx.custom_function(forward)
affine_scan.vjp(backward)
