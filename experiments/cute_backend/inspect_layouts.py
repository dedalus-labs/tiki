"""Use the monorepo Tiki visualizer to check thread/value and bank mappings."""

import argparse
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import cutlass
import matplotlib
from cutlass import cute

import tiki as tk

matplotlib.use("Agg")

ThreadValueLayout = tuple[tuple[tuple[int, int], int], tuple[tuple[int, int], int]]


@dataclass(frozen=True)
class SharedLayout:
    swizzle: tk.Swizzle
    shape: tuple[int, int] = (32, 32)

    def __getitem__(self, coordinate: tuple[int, int]) -> int:
        row, col = coordinate
        return self.swizzle(row * 32 + col)


def positions(tv: ThreadValueLayout) -> list[list[int]]:
    shape, stride = tv
    threads = math.prod(shape[0])
    values = shape[1]

    @cute.jit
    def query():
        layout = cute.make_layout(shape, stride=stride)
        output = []
        for thread in cutlass.range_constexpr(threads):
            row = []
            for value in cutlass.range_constexpr(values):
                row.append(layout((thread, value)))
            output.append(row)
        return output

    return query()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("monorepo_debug", args.visualizer)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load visualizer: {args.visualizer}")
    debug = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(debug)
    import matplotlib.pyplot as plt

    schedule = tk.TransposeSchedule()
    warps = schedule.threads // 32
    values = 1024 // schedule.threads
    layouts = {
        "transpose_load": (((32, warps), values), ((32, 1), warps)),
        "transpose_read": (((32, warps), values), ((1, 32), schedule.threads)),
    }
    for name, tv in layouts.items():
        offsets = positions(tv)
        for thread in range(schedule.threads):
            for value in range(values):
                row, col = thread // 32 + value * warps, thread % 32
                if name == "transpose_read":
                    row, col = col, row
                assert offsets[thread][value] == row + col * 32
        fig, ax = debug.visualize_tv(
            tile=(32, 32),
            tv=tv,
            path=None,
            font_size=5,
            cell_px=30,
            color_fn=lambda thread, value: plt.cm.Set2.colors[thread // 32],
        )
        ax.set_title(f"{name}: logical shared-memory coordinates (T=thread, V=value)")
        fig.savefig(args.output / f"{name}.svg", bbox_inches="tight")
        fig.savefig(args.output / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
    for name, swizzle in (
        ("banks_plain", tk.Swizzle(0, 0, 5)),
        ("banks_xor", tk.Swizzle(5, 0, 5)),
    ):
        fig, ax = debug.visualize_layout(
            SharedLayout(swizzle),
            figsize=(11, 11),
            color_map=lambda offset: plt.cm.hsv((offset % 32) / 32)[:3],
            label_map=lambda offset: str(offset % 32),
        )
        ax.set_title(
            f"{name}: shared-memory bank per logical coordinate (32-bit words)"
        )
        fig.savefig(args.output / f"{name}.svg", bbox_inches="tight")
        fig.savefig(args.output / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
    print("CuTe TV layouts match every emitted transpose thread/value coordinate")


if __name__ == "__main__":
    main()
