# Copyright © 2025 Apple Inc.

import argparse
import time

import numpy as np
import tiki as tk

TIKI_DTYPES = {
    "float16": tk.float16,
    "bfloat16": tk.bfloat16,
    "float32": tk.float32,
}


def parse_cases(cases):
    parsed = []
    for spec in cases.split(","):
        parts = spec.split("x")
        m, n, k, bs = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        sparsity = float(parts[4]) if len(parts) > 4 else 0.5
        parsed.append((m, n, k, bs, sparsity))
    return parsed


def make_masks(m, n, k, block_size, sparsity, rng):
    """Create block masks with given sparsity (fraction of blocks zeroed)."""
    tm = (m + block_size - 1) // block_size
    tn = (n + block_size - 1) // block_size
    tk = (k + block_size - 1) // block_size

    lhs_mask = (rng.random((tm, tk)) >= sparsity).astype(np.bool_)
    rhs_mask = (rng.random((tk, tn)) >= sparsity).astype(np.bool_)
    out_mask = (rng.random((tm, tn)) >= sparsity).astype(np.bool_)
    return lhs_mask, rhs_mask, out_mask


def tiki_naive_block_masked_mm(a, b, block_size, out_mask, lhs_mask, rhs_mask):
    """Tiki naive: expand masks and use regular matmul."""
    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-1]

    def expand(mask, rows, cols):
        e = tk.repeat(tk.repeat(mask, block_size, axis=-2), block_size, axis=-1)
        return e[..., :rows, :cols]

    a_masked = a * expand(lhs_mask, M, K)
    b_masked = b * expand(rhs_mask, K, N)
    c = a_masked @ b_masked
    c = c * expand(out_mask, M, N)
    return c


def bench_tiki(fn, warmup, iters):
    for _ in range(warmup):
        y = fn()
        tk.eval(y)
    tk.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        tk.eval(y)
    tk.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row):
        return (
            "| "
            + " | ".join(f"{cell:<{widths[i]}}" for i, cell in enumerate(row))
            + " |"
        )

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark block_masked_mm vs naive expand+matmul"
    )
    parser.add_argument(
        "--cases",
        default=(
            "256x256x256x32x0.5,"
            "512x512x512x32x0.5,"
            "1024x1024x1024x32x0.5,"
            "1024x1024x1024x64x0.5,"
            "2048x2048x2048x64x0.5,"
            "256x256x256x32x0.0,"
            "1024x1024x1024x32x0.0,"
            "1024x1024x1024x32x0.9"
        ),
        help="Comma-separated MxNxKxBSxSparsity list. Sparsity=fraction of blocks zeroed.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    tiki_dtype = TIKI_DTYPES[args.dtype]

    print(f"dtype={args.dtype}  warmup={args.warmup}  iters={args.iters}")

    headers = [
        "Case (MxNxKxBS)",
        "Sparsity",
        "Tiki ms",
        "Naive ms",
        "Speedup",
    ]
    if not args.no_check:
        headers.append("Max err")
    rows = []

    cases = parse_cases(args.cases)
    for idx, (m, n, k, bs, sparsity) in enumerate(cases):
        rng = np.random.default_rng(args.seed + idx)
        a_np = rng.standard_normal((m, k)).astype(np.float32)
        b_np = rng.standard_normal((k, n)).astype(np.float32)
        lhs_mask_np, rhs_mask_np, out_mask_np = make_masks(m, n, k, bs, sparsity, rng)

        a_mx = tk.array(a_np, dtype=tiki_dtype)
        b_mx = tk.array(b_np, dtype=tiki_dtype)
        lhs_mask_mx = tk.array(lhs_mask_np)
        rhs_mask_mx = tk.array(rhs_mask_np)
        out_mask_mx = tk.array(out_mask_np)
        tk.eval(a_mx, b_mx, lhs_mask_mx, rhs_mask_mx, out_mask_mx)

        # Correctness check: block_masked_mm vs naive expand+matmul
        err_str = ""
        if not args.no_check:
            y_op = tk.block_masked_mm(
                a_mx, b_mx, bs, out_mask_mx, lhs_mask_mx, rhs_mask_mx
            )
            y_naive = tiki_naive_block_masked_mm(
                a_mx, b_mx, bs, out_mask_mx, lhs_mask_mx, rhs_mask_mx
            )
            tk.eval(y_op, y_naive)
            err = float(tk.max(tk.abs(y_op - y_naive)).item())
            err_str = f"{err:.2e}"

        # Benchmark
        t_tiki = bench_tiki(
            lambda: tk.block_masked_mm(
                a_mx, b_mx, bs, out_mask_mx, lhs_mask_mx, rhs_mask_mx
            ),
            args.warmup,
            args.iters,
        )
        t_naive = bench_tiki(
            lambda: tiki_naive_block_masked_mm(
                a_mx, b_mx, bs, out_mask_mx, lhs_mask_mx, rhs_mask_mx
            ),
            args.warmup,
            args.iters,
        )
        speedup = f"{t_naive / t_tiki:.2f}x" if t_tiki > 0 else "-"

        row = [
            f"{m}x{n}x{k}x{bs}",
            f"{sparsity:.0%}",
            f"{t_tiki:.3f}",
            f"{t_naive:.3f}",
            speedup,
        ]
        if not args.no_check:
            row.append(err_str)
        rows.append(row)

    print_table(headers, rows)
    if not args.no_check:
        print("err: max|block_masked_mm - naive_expand_matmul|")


if __name__ == "__main__":
    main()
