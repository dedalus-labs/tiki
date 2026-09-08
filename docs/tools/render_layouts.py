# Copyright © 2026 Dedalus Labs, Inc.

"""Render the layout maps in the native layouts page with the Tiki visualizer.

The visualizer is ``tiki.kernels.cute.lib.debug`` from the Dedalus monorepo,
loaded by path as ``experiments/cute_backend/inspect_layouts.py`` does::

    python docs/tools/render_layouts.py --visualizer PATH/TO/debug.py \\
        --output docs/src/_static/layouts
"""

import argparse
import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path

import matplotlib
import mlx.tiki as tk

matplotlib.use("Agg")


class Grid:
    """The visualizer reads ``shape`` and ``layout[row, column]``."""

    def __init__(self, layout: tk.Layout | tk.ComposedLayout) -> None:
        self.layout = layout
        self.shape = layout.shape

    def __getitem__(self, coordinate: tuple[int, int]) -> int:
        return self.layout(*coordinate)


def load_visualizer(path: Path) -> types.ModuleType:
    """The module imports CuTe for its kernel helpers; the grid renderer does not use it."""
    cute = types.ModuleType("cutlass.cute")
    cute.__getattr__ = lambda name: (lambda *args, **kwargs: None)  # type: ignore[attr-defined]
    cutlass = types.ModuleType("cutlass")
    cutlass.cute = cute  # type: ignore[attr-defined]
    sys.modules.setdefault("cutlass", cutlass)
    sys.modules.setdefault("cutlass.cute", cute)
    spec = importlib.util.spec_from_file_location("tiki_debug", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load visualizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    debug = load_visualizer(args.visualizer)
    import matplotlib.pyplot as plt

    tints = [plt.cm.tab20c.colors[index] for index in (3, 7, 11, 15)]
    by_storage_row: Callable[[int], tuple[float, ...]] = lambda index: tints[index // 4]
    by_tile: Callable[[int], tuple[float, ...]] = lambda index: tints[
        (index // 4 // 2) * 2 + (index % 4) // 2
    ]
    base = tk.Layout((4, 4), stride=(4, 1))
    figures = {
        "layout-row-major": (base, by_storage_row),
        "layout-column-major": (tk.Layout((4, 4), stride=(1, 4)), by_storage_row),
        "layout-tiled": (tk.logical_divide(base, (2, 2)), by_tile),
        "layout-swizzled": (
            base.swizzle(tk.Swizzle(bits=2, base=0, shift=2)),
            by_storage_row,
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (layout, color_map) in figures.items():
        figure, _ = debug.visualize_layout(
            Grid(layout), color_map=color_map, figsize=(3.2, 3.2)
        )
        figure.savefig(args.output / f"{name}.svg", format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
