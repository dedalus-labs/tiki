from itertools import product

import tiki as tk


# In mxfp8 mode, the results do not match exactly:
# fewer than 1% of output elements differ.
# This does not appear to be a systematic error.
# The error can exceed 1 ULP for very small values,
# and is always below 1 ULP for larger values.
# For nvfp4, the results match exactly.
# therefore I suspect that the discrepancy comes from
# the mxfp8 matmul implementation in cuBLASLt..
def ulp_bf16_at(x):
    ax = tk.abs(x)
    min_normal = tk.array(2.0**-126)
    ax = tk.where(ax < min_normal, min_normal, ax)
    e = tk.floor(tk.log2(ax))
    return tk.power(2.0, e - 7.0)


def test_qqmm():
    key = tk.random.key(0)
    k1, k2 = tk.random.split(key)
    dtypes = [tk.bfloat16, tk.float32, tk.float16]

    tests = (
        (16, "nvfp4", 4),
        (32, "mxfp8", 8),
    )
    shapes = (
        [64, 65, 33, 128, 256, 1024, 1024 * 8],  # M
        [64, 128, 256, 1024, 1024 * 8],  # N
        [64, 128, 256, 1024, 1024 * 8],  # K
    )
    layouts = ["TN", "NT", "TT", "NN"]
    for group_size, mode, bits in tests:
        for M, N, K in product(*shapes):
            for dtype in dtypes:
                for layout in layouts:
                    if layout == "NT":
                        x_shape = (M, K)
                        w_shape = (N, K)
                    elif layout == "TN":
                        x_shape = (K, M)
                        w_shape = (K, N)
                    elif layout == "TT":
                        x_shape = (K, M)
                        w_shape = (N, K)
                    else:  # "NN"
                        x_shape = (M, K)
                        w_shape = (K, N)

                    x = tk.random.normal(shape=x_shape, key=k1, dtype=dtype)
                    w = tk.random.normal(shape=w_shape, key=k2, dtype=dtype)

                    if layout == "TT":
                        x = tk.transpose(x)
                    elif layout == "TN":
                        w = tk.transpose(w)
                        x = tk.transpose(x)
                    elif layout == "NN":
                        w = tk.transpose(w)

                    y_q = tk.qqmm(
                        x,
                        w,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )
                    w_q, scales_w = tk.quantize(w, group_size, bits, mode=mode)
                    w_dq = tk.dequantize(
                        w_q,
                        scales_w,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                        dtype=dtype,
                    )
                    x_q, scales_x = tk.quantize(
                        x, group_size=group_size, bits=bits, mode=mode
                    )
                    x_dq = tk.dequantize(
                        x_q,
                        scales_x,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                        dtype=dtype,
                    )
                    y_hat = tk.matmul(x_dq, tk.transpose(w_dq))
                    ulp = ulp_bf16_at(y_hat)
                    error = (y_q - y_hat).abs()
                    if not (tk.logical_or(error < 1e-3, error <= ulp).all()):
                        raise AssertionError(
                            f"qqmm test failed for shape {(M, N, K)}, "
                            f"group_size={group_size}, bits={bits}, "
                            f"mode={mode}, dtype={dtype}, layout={layout}"
                        )


def test_qqmm_vjp():
    key = tk.random.key(0)
    k1, k2 = tk.random.split(key)
    M = 64
    N = 1024
    K = 512
    tests = (
        (16, "nvfp4", 4),
        (32, "mxfp8", 8),
    )
    x = tk.random.normal(shape=(M, K), key=k1)
    c = tk.ones(shape=(M, N))

    for group_size, mode, bits in tests:
        w = tk.random.normal(shape=(N, K), key=k2)

        def fn(x):
            return tk.qqmm(x, w, group_size=group_size, bits=bits, mode=mode)

        _, vjp_out = tk.vjp(fn, primals=(x,), cotangents=(c,))
        w_tq, scales_wt = tk.quantize(
            tk.transpose(w), group_size=group_size, bits=bits, mode=mode
        )
        expected_out = tk.qqmm(
            c, w_tq, scales_wt, group_size=group_size, bits=bits, mode=mode
        )
        ulp = ulp_bf16_at(expected_out)
        error = (vjp_out[0] - expected_out).abs()
        if not (tk.logical_or(error < 1e-3, error <= ulp).all()):
            raise AssertionError(
                f"qqmm vjp test failed for shape {(M, N, K)}, "
                f"group_size={group_size}, bits={bits}, mode={mode}"
            )


if __name__ == "__main__":
    test_qqmm()
    test_qqmm_vjp()
