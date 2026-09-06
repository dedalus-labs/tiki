# CUDA allocator validation: September 6, 2026

The Rust allocator passes the scoped copy-before-free tests. These results do
not establish memory safety for all CUDA execution or graph replay paths.

## Scope and results

The migration path submits the device-to-host copy and device-source release
on one stream. Blocking export waits for that copy. The tests either finish
producer work before export or submit the producer and migration on the same
stream. Ownership of work submitted on other streams remains outside this
allocator slice.

Independent builds of the initial port (`9191de8d`) and atomic address lookup
change (`1e68053f`) each pass all three
[GPU allocator tests](../../mlx/backend/cuda/runtime/tests/forced_reuse.rs)
under Compute Sanitizer memcheck, with zero reported errors. The tests check
forced address reuse after blocking export, stream-ordered export, and cache
accounting. They run serially because they share the global allocator.

The available C++ and Rust MLX builds also pass the memory and array suites:
107 tests run, 87 passed, and 20 skipped on each build. This includes 84 passed
and 19 skipped in `test_array`, not 103 passed plus 19 skipped. The recorded
`test_ops` results have 159 passed, three errors, and one skipped on each
build. All three error sites explicitly request a CPU stream in builds with
`MLX_BUILD_CPU=OFF`; these results do not qualify a CPU-enabled build.

The separate reports from exports without an explicit producer synchronization
remain outside this qualification. This record does not attribute them to
either an execution-order defect or sanitizer event-query tracking, and does
not claim that a proposed event-wait change resolves them.

## Reproduce the allocator checks

Run from the repository root on a GH200 with CUDA 13.3 and Rust 1.92 or later.
Use a fresh build directory when testing an archived source revision.

```sh
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CARGO_TARGET_DIR=$(mktemp -d)
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="$CUDA_TOOLKIT_PATH/bin/compute-sanitizer --tool memcheck --error-exitcode 99"
cargo test --locked --release \
  --manifest-path mlx/backend/cuda/runtime/Cargo.toml \
  --target aarch64-unknown-linux-gnu --test forced_reuse \
  -- --ignored --test-threads=1
```

Expected result: three passed tests and `ERROR SUMMARY: 0 errors`. A normal
Cargo test run without the sanitizer runner is a separate correctness check.

## Host-export measurements

The [host-export benchmark](repros/host_export.py) separates a warmed allocation
cache from fresh device storage. It disables and clears the MLX cache for the
fresh-storage measurements. Each group uses five warm-up iterations followed
by 30 measured iterations, with cache-enabled and cache-disabled groups in
ABBA order. The timed operation is `np.asarray`; GPU production and its
synchronization occur before the timer. Export includes any host-storage
allocation, copy, and synchronization performed by that operation.

Run the same script in each CUDA build's Python environment:

```sh
python experiments/rust_backend/repros/host_export.py
```

The GH200 measurements below are ranges of the two group medians, in
microseconds. They compare the supplied C++ and Rust binary artifacts, not a
new pair of full MLX builds from the source revisions above.

| Export | C++ | Rust |
| --- | ---: | ---: |
| 1 MiB, warmed cache | 2.24–2.30 | 2.24–2.38 |
| 1 MiB, fresh device storage | 20.67–21.70 | 19.34–20.45 |
| 64 MiB, warmed cache | 2.18–2.35 | 1.62–2.13 |
| 64 MiB, fresh device storage | 1261–1297 | 1322–1324 |

Warmed cache timings do not measure device-to-host migration latency. This
short run does not establish a statistically significant performance
difference between implementations.

The earlier tiny-kernel dispatch comparison is preliminary. The address
lookup change replaces a mutex read with an atomic load; inlining alone does
not remove a mutex. Its effect needs an isolated before-and-after benchmark.
No result here establishes that its overhead disappears for larger kernels.
