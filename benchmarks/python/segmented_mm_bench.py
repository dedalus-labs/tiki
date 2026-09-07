# Copyright © 2026 Apple Inc.

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
        m, n, k, s = [int(x) for x in spec.split("x")]
        parsed.append((m, n, k, s))
    return parsed


def make_segments(k, num_segments, pattern, seed):
    if pattern == "equal":
        cuts = np.linspace(0, k, num_segments + 1, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        cuts = rng.integers(0, k + 1, size=(num_segments - 1,), dtype=np.int64)
        cuts = np.sort(cuts)
        cuts = np.concatenate(([0], cuts, [k]))
    return np.stack([cuts[:-1], cuts[1:]], axis=1).astype(np.uint32)


def numpy_segmented_mm_ref(a, b, segments):
    """Ground-truth reference in float64."""
    out = []
    for start, end in segments:
        out.append(a[:, start:end] @ b[start:end, :])
    return np.stack(out, axis=0)


def tiki_segmented_mm_loop(a, b, segments):
    """Tiki loop-of-matmuls baseline."""
    segments_list = segments.tolist()
    out = []
    for start, end in segments_list:
        out.append(a[:, start:end] @ b[start:end, :])
    return tk.stack(out, axis=0)


def bench_tiki(a, b, segments, warmup, iters):
    for _ in range(warmup):
        y = tk.segmented_mm(a, b, segments)
        tk.eval(y)
    tk.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        y = tk.segmented_mm(a, b, segments)
        tk.eval(y)
    tk.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e3 / iters


def bench_tiki_loop(a, b, segments, warmup, iters):
    for _ in range(warmup):
        y = tiki_segmented_mm_loop(a, b, segments)
        tk.eval(y)
    tk.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        y = tiki_segmented_mm_loop(a, b, segments)
        tk.eval(y)
    tk.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e3 / iters


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=(
            "128x128x1024x16,"
            "128x128x1024x32,"
            "256x256x2048x16,"
            "512x512x4096x32,"
            "1024x1024x4096x32,"
            "1024x1024x8192x64"
        ),
        help="Comma-separated MxNxKxS list.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--segments",
        choices=["equal", "random"],
        default="random",
        help="Segment generation pattern.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    tiki_dtype = TIKI_DTYPES[args.dtype]

    print(
        f"dtype={args.dtype} warmup={args.warmup} iters={args.iters} segments={args.segments}"
    )

    headers = [
        "Case",
        "Tiki ms",
        "Loop ms",
        "Speedup",
        "Tiki err",
        "Loop err",
    ]
    rows = []

    cases = parse_cases(args.cases)
    for idx, (m, n, k, s) in enumerate(cases):
        rng = np.random.default_rng(args.seed + idx)
        a_np = rng.standard_normal((m, k)).astype(np.float32)
        b_np = rng.standard_normal((k, n)).astype(np.float32)
        seg_np = make_segments(k, s, args.segments, args.seed + idx)

        a_mx = tk.array(a_np, dtype=tiki_dtype)
        b_mx = tk.array(b_np, dtype=tiki_dtype)
        seg_mx = tk.array(seg_np, dtype=tk.uint32)
        tk.eval(a_mx, b_mx, seg_mx)

        tiki_err_str = ""
        loop_err_str = ""
        if not args.no_check:
            y_tiki = tk.segmented_mm(a_mx, b_mx, seg_mx)
            y_loop = tiki_segmented_mm_loop(a_mx, b_mx, seg_mx)
            tk.eval(y_tiki, y_loop)

            if args.dtype == "float32":
                ref = numpy_segmented_mm_ref(
                    a_np.astype(np.float64),
                    b_np.astype(np.float64),
                    seg_np.tolist(),
                )
                tiki_err = np.max(np.abs(np.array(y_tiki, dtype=np.float64) - ref))
                loop_err = np.max(np.abs(np.array(y_loop, dtype=np.float64) - ref))
            else:
                a_mx_f32 = tk.array(a_np, dtype=tk.float32)
                b_mx_f32 = tk.array(b_np, dtype=tk.float32)
                ref = tk.segmented_mm(a_mx_f32, b_mx_f32, seg_mx)
                tk.eval(ref)
                tiki_err = float(tk.max(tk.abs(ref - y_tiki.astype(tk.float32))).item())
                loop_err = float(tk.max(tk.abs(ref - y_loop.astype(tk.float32))).item())
            tiki_err_str = f"{tiki_err:.2e}"
            loop_err_str = f"{loop_err:.2e}"

        t_tiki = bench_tiki(a_mx, b_mx, seg_mx, args.warmup, args.iters)
        t_loop = bench_tiki_loop(a_mx, b_mx, seg_mx, args.warmup, args.iters)
        ratio = t_loop / t_tiki if t_tiki > 0 else float("inf")
        rows.append(
            [
                f"{m}x{n}x{k}x{s}",
                f"{t_tiki:.3f}",
                f"{t_loop:.3f}",
                f"{ratio:.2f}x",
                tiki_err_str,
                loop_err_str,
            ]
        )

    print_table(headers, rows)
    if not args.no_check:
        if args.dtype == "float32":
            print("err: max|result - numpy_fp64_ref|")
        else:
            print("err: max|result - own_fp32_result|")


if __name__ == "__main__":
    main()
