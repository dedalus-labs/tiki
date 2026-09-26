# Associative scans

This experiment provides a generic MLX scan and a CUDA kernel for affine pairs.
The generic algorithm follows
[`jax.lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html).

## Generic scan

`associative_scan(fn, elems, reverse=False, axis=0)` combines adjacent pairs,
recursively scans the reduced sequence, and interleaves the remaining prefixes.
It preserves the axis order presented to `fn`, container types supported by
MLX's tree mapper, and dictionary key identity. Dictionary insertion order does
not assign leaves to different fields. The combine must preserve tree structure,
leaf shapes, and dtypes, and must act independently along the scan axis.

The caller supplies an associative operation. Float32 reassociation can change
rounding. Tests use stated tolerances, not a claim of bitwise equality with JAX.
The generic implementation uses MLX operations, so its derivatives come from
MLX. Strided-slice differentiation and vectorization require the singleton
slice-normalization fix. The tests keep that dependency separate from forward
parity tests.

## CUDA affine scan

`affine_scan(a, b)` accepts matching float32 arrays of shape `[batch, time]` with
positive batch and `1 <= time <= 2048`. The launch grid must fit MLX's signed
32-bit grid fields. The kernel uses wide row offsets and reports contract
violations before launch. Inputs are packed by MLX's CUDA-kernel interface when
necessary.

The affine combine is:

```text
(a_left, b_left) * (a_right, b_right)
    = (a_right * a_left, a_right * b_left + b_right)
```

The CUDA kernel supplies a registered reverse-mode VJP. It does not supply a
JVP or a higher-order derivative rule for its backward kernel. The empty prefix
is handled structurally. Multiplying an infinite coefficient by an artificial
zero would change the first offset and is not a valid implementation of that
prefix.

## Verification and benchmarks

```sh
cd experiments/associative_scan
python -m unittest test_scan test_scan_structure test_affine_scan
python bench_affine_scan.py --tree eager
```

The benchmark measures warm host dispatch plus blocking evaluation. Both VJP
columns include required forward work. It does not compare a full tree VJP
against a kernel given precomputed forward outputs.

`--tree compiled` explicitly requests a compiled generic baseline. A compiler
failure terminates that run. It never switches to the eager baseline. The
axis-1 affine-tree program still fails in `cuGraphAddKernelNode` on the
reviewed MLX CUDA build, including after the tree and axis fixes. These timings
are integration evidence, not a
comparison with CUB or an optimized CuTe scan.
