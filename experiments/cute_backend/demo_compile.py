"""Write the schedule and CuTe MLIR for an ordinary MLX function."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

import tiki as tk


@tk.compile(backend="cute", schedule=tk.Schedule(threads=128, elements_per_thread=4))
def affine(x: mx.array, y: mx.array) -> mx.array:
    return x * y + 2.0 - y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    x = mx.arange(513, dtype=mx.float32)
    y = mx.array(3.0)
    lowered = affine.lower(x, y)
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = {**asdict(lowered.schedule), "grid_threads": lowered.grid}
    (args.output / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    (args.output / "graph.json").write_text(
        json.dumps(asdict(lowered.graph), indent=2) + "\n"
    )
    (args.output / "kernel.mlir").write_text(lowered.mlir)
    print(json.dumps(schedule))
    if args.execute:
        result = affine(x, y)
        error = mx.max(mx.abs(result - (x * y + 2.0 - y))).item()
        if error != 0:
            raise AssertionError(f"CUDA result error: {error}")
        print(
            json.dumps(
                {"device": mx.device_info(mx.gpu)["device_name"], "max_error": error}
            )
        )


if __name__ == "__main__":
    main()
