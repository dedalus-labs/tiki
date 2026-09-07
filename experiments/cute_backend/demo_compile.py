"""Write the schedule and CuTe MLIR for an ordinary Tiki function."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import tiki as tk

import compiler


@compiler.compile(backend="cute", schedule=compiler.Schedule(threads=128, elements_per_thread=4))
def affine(x: tk.array, y: tk.array) -> tk.array:
    return x * y + 2.0 - y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    x = tk.arange(513, dtype=tk.float32)
    y = tk.array(3.0)
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
        error = tk.max(tk.abs(result - (x * y + 2.0 - y))).item()
        if error != 0:
            raise AssertionError(f"CUDA result error: {error}")
        print(
            json.dumps(
                {"device": tk.device_info(tk.gpu)["device_name"], "max_error": error}
            )
        )


if __name__ == "__main__":
    main()
