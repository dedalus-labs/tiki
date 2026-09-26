# Copyright © 2026 Dedalus Labs, Inc.

"""Compare cached host views with fresh device-to-host exports."""

import statistics
import time

import mlx.core as mx
import numpy as np

WARMUP = 5
SAMPLES = 30


def measure_export(elements: int, cache_bytes: int) -> float:
    """Measure export latency after producer completion.

    A zero cache limit prevents prior exports from supplying host-visible
    storage to the next GPU result. GPU production and its allocation are not
    timed; host-storage allocation during export is timed.
    """
    previous_limit = mx.set_cache_limit(cache_bytes)
    samples: list[int] = []
    try:
        mx.clear_cache()
        source = mx.arange(elements, dtype=mx.float32)
        mx.eval(source)
        mx.synchronize()
        for iteration in range(WARMUP + SAMPLES):
            result = source + 1
            mx.eval(result)
            mx.synchronize()
            start = time.perf_counter_ns()
            exported = np.asarray(result)
            elapsed = time.perf_counter_ns() - start
            assert exported[0] == 1
            assert exported[-1] == elements
            if iteration >= WARMUP:
                samples.append(elapsed)
            del exported, result
        return statistics.median(samples) / 1_000
    finally:
        mx.set_cache_limit(previous_limit)
        mx.clear_cache()


def main() -> None:
    if not mx.cuda.is_available():
        raise RuntimeError("host export benchmark requires MLX CUDA")
    mx.set_default_device(mx.gpu)
    print(f"module={mx.__file__}")
    for elements in (262_144, 16_777_216):
        for cache_bytes in (1 << 30, 0, 0, 1 << 30):
            latency_us = measure_export(elements, cache_bytes)
            print(
                f"elements={elements} cache_bytes={cache_bytes} "
                f"median_us={latency_us:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
