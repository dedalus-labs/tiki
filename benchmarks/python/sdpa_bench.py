# Copyright © 2024 Apple Inc.

import argparse
import math
import os
import subprocess
import time

import tiki as tk
import numpy as np

device_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
device_name = device_name.decode("utf-8").strip("\n")

N_warmup = 5
N_iter_bench = 40
N_iter_func = 8


def bench(f, *args):
    for i in range(N_warmup):
        f(*args)

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(*args)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def prepare_inputs(B, qL, kL, D, qH, kH, mask, transpose, dtype):
    np_dtype = getattr(np, dtype)

    shape_q = (B, qL, qH, D) if transpose else (B, qH, qL, D)
    shape_kv = (B, kL, kH, D) if transpose else (B, kH, kL, D)

    scale = 1.0 / math.sqrt(D)

    q_np = np.random.normal(0.0, 1.0, shape_q).astype(np_dtype)
    k_np = np.random.normal(0.0, scale, shape_kv).astype(np_dtype)
    v_np = np.random.normal(0.0, scale, shape_kv).astype(np_dtype)

    q_mx = tk.array(q_np)
    k_mx = tk.array(k_np)
    v_mx = tk.array(v_np)

    if mask is not None:
        if mask == "additive":
            mask_np = np.random.normal(0.0, 1.0, (B, qH, qL, kL)).astype(np_dtype)
            mask = tk.array(mask_np)
        elif mask == "bool":
            mask_np = np.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5
            mask = tk.array(mask_np)

    return q_mx, k_mx, v_mx, scale, mask


def tiki_ref_attn(q, k, v, scale=1.0, mask=None):
    q_dtype = q.dtype
    q = q * tk.array(scale, q_dtype)
    n_q_heads = q.shape[-3]
    n_kv_heads = k.shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    B = q.shape[0]
    L = q.shape[2]
    kL = k.shape[2]

    if n_repeats > 1:
        q = tk.reshape(q, [B, n_kv_heads, n_repeats, L, -1])
        k = tk.expand_dims(k, 2)
        v = tk.expand_dims(v, 2)

    scores = q @ tk.swapaxes(k, -1, -2)

    if mask is not None:

        if mask == "causal":
            q_offset = max(0, kL - L)
            q_indices = tk.arange(q_offset, q_offset + L)
            k_indices = tk.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]

        if n_repeats > 1 and mask.ndim >= 3:
            if mask.shape[-3] == 1:
                mask = tk.expand_dims(mask, -3)
            else:
                mask = tk.unflatten(mask, -3, (n_kv_heads, n_repeats))

        if mask.dtype == tk.bool_:
            scores = tk.where(mask, scores, -np.float32(np.inf))
        else:
            scores += mask

    scores = tk.softmax(scores, axis=-1, precise=True)

    out = scores @ v
    if n_repeats > 1:
        out = tk.reshape(out, [B, n_q_heads, L, -1])

    return out


def tiki_fused_attn(q, k, v, scale, mask):
    return tk.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def do_attention(f, q, k, v, scale, mask=None, transpose=False):
    if transpose:
        q_t = tk.transpose(q, (0, 2, 1, 3))
        k_t = tk.transpose(k, (0, 2, 1, 3))
        v_t = tk.transpose(v, (0, 2, 1, 3))
        o_t = f(q_t, k_t, v_t, scale=scale, mask=mask)
        return tk.transpose(o_t, (0, 2, 1, 3))
    else:
        return f(q, k, v, scale=scale, mask=mask)


def do_attention_bench(f, q, k, v, scale, mask=None, transpose=False):
    q_out = q

    for i in range(N_iter_func):
        q_out = do_attention(f, q_out, k, v, scale, mask=mask, transpose=transpose)

    tk.eval(q_out)
    return q_out


def bench_shape(
    B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype, transpose=True, mask_in=None
):
    q_mx, k_mx, v_mx, scale, mask = prepare_inputs(
        B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_in, transpose, dtype
    )

    time_tiki_unfused = bench(
        do_attention_bench, tiki_ref_attn, q_mx, k_mx, v_mx, scale, mask, transpose
    )
    time_tiki_fused = bench(
        do_attention_bench, tiki_fused_attn, q_mx, k_mx, v_mx, scale, mask, transpose
    )

    o_tiki_fused = do_attention(tiki_ref_attn, q_mx, k_mx, v_mx, scale, mask, transpose)
    o_tiki_unfused = do_attention(
        tiki_fused_attn, q_mx, k_mx, v_mx, scale, mask, transpose
    )

    atol = 1e-5 if dtype == "float32" else 2e-4

    if not tk.allclose(o_tiki_fused, o_tiki_unfused, atol=atol, rtol=atol):
        print(
            f"Failed at (B: {B}, qsl: {qsl}, ksl: {ksl}, head_dim: {head_dim}, n_qh: {n_q_heads}, n_kvh: {n_kv_heads}, mask: {mask_in}) [tpose = {transpose}] with max(|a - b|) = {tk.max(tk.abs(o_tiki_unfused - o_tiki_fused)):3.2e}"
        )

    return time_tiki_fused, time_tiki_unfused


def get_gflop_count(B, M, N, K):
    return float(2.0 * N_iter_bench * N_iter_func * B * M * N * K) / float(1024.0**3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gemm benchmarks")

    dtypes = ("float16", "float32")[:1]
    transposes = (False,)

    # fmt: off
    shapes_64 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,    32,    32,       64,   32,    32),
          (  1,    64,    64,       64,   32,    32),
          (  1,   128,   128,       64,   32,    32),
          (  1,   256,   256,       64,   32,    32),
          (  1,   512,   512,       64,   32,    32),
          (  1,  1024,  1024,       64,   32,     8),
          (  1,  2048,  2048,       64,   32,     8),
          (  1,  4096,  4096,       64,   32,     8),
          (  1,  4096,  5000,       64,   32,     8),
          (  1,  2048,  32121,      64,   32,     8),
    )

    shapes_72 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,  1024,  1024,       72,   32,     8),
          (  1,  2048,  2048,       72,   32,     8),
          (  1,  4096,  4096,       72,   32,     8),
          (  1,  4096,  5000,       72,   32,     8),
          (  1,  2048,  32121,      72,   32,     8),
    )

    shapes_80 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,  1024,  1024,       80,   32,     8),
          (  1,  2048,  2048,       80,   32,     8),
          (  1,  4096,  4096,       80,   32,     8),
          (  1,  4096,  5000,       80,   32,     8),
          (  1,  2048,  32121,      80,   32,     8),
    )

    shapes_96 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,  1024,  1024,       96,   32,     8),
          (  1,  2048,  2048,       96,   32,     8),
          (  1,  4096,  4096,       96,   32,     8),
          (  1,  4096,  5000,       96,   32,     8),
          (  1,  2048,  32121,      96,   32,     8),
    )

    shapes_128 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,  1024,  1024,      128,   32,     8),
          (  1,  2048,  2048,      128,   32,     8),
          (  1,  4096,  4096,      128,   32,     8),
          (  1,  4096,  5000,      128,   32,     8),
          (  1,  2048,  32121,     128,   32,     8),
    )

    shapes_256 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,  1024,  1024,      256,   24,     4),
          (  1,  2048,  2048,      256,   24,     4),
          (  1,  4096,  4096,      256,   24,     4),
          (  1,  4096,  5000,      256,   24,     4),
          (  1,  2048,  32121,     256,   24,     4),
    )
    # fmt: on

    shapes = shapes_64 + shapes_72 + shapes_80 + shapes_96 + shapes_128 + shapes_256

    masks = [None, "bool", "causal"]

    print(
        "  B,   qsl,   ksl, hdim, n_qh, n_kvh, t,   dtype,     mask, t_unfs, t_fuse, diff%"
    )

    for dtype in dtypes:
        for transpose in transposes:
            for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in shapes:
                for mask_in in masks:
                    time_tiki_fused, time_tiki_unfused = bench_shape(
                        B,
                        qsl,
                        ksl,
                        head_dim,
                        n_q_heads,
                        n_kv_heads,
                        dtype,
                        transpose,
                        mask_in,
                    )
                    diff = time_tiki_unfused / time_tiki_fused - 1.0
                    t_str = 1 if transpose else 0
                    print(
                        f"{B:3d}, {qsl:5d}, {ksl:5d}, {head_dim:4d}, {n_q_heads:4d}, {n_kv_heads:5d}, {t_str:1d}, {dtype}, {str(mask_in):>8}, {time_tiki_unfused: 2.3f}, {time_tiki_fused: 2.3f}, {100. * diff:+5.2f}%"
                    )
