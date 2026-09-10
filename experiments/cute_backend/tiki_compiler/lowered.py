"""A generated tensor kernel and the launch metadata derived from its schedule."""

from dataclasses import dataclass
from math import prod

from .graph import Graph, Shape
from .schedule import RowSchedule, Schedule, TransposeSchedule


@dataclass(frozen=True, kw_only=True)
class Lowered:
    graph: Graph
    schedule: Schedule | RowSchedule | TransposeSchedule
    mlir: str

    @property
    def grid(self) -> tuple[int, int, int]:
        if isinstance(self.schedule, RowSchedule):
            rows = self.graph.shape[0]
            blocks = (rows + self.schedule.rows_per_block - 1) // self.schedule.rows_per_block
            grid = (blocks * self.schedule.threads, 1, 1)
            return grid
        if isinstance(self.schedule, TransposeSchedule):
            rows, cols = self.graph.shape
            blocks = ((rows + 31) // 32) * ((cols + 31) // 32)
            grid = (blocks * self.schedule.threads, 1, 1)
            return grid
        tile = self.schedule.threads * self.schedule.elements_per_thread
        blocks = (prod(self.graph.shape) + tile - 1) // tile
        grid = (blocks * self.schedule.threads, 1, 1)
        return grid

    @property
    def output_shapes(self) -> tuple[Shape, ...]:
        result = tuple(value.shape for value in self.graph.outputs)
        return result

    @property
    def shared_memory_bytes(self) -> int:
        if isinstance(self.schedule, RowSchedule):
            if not any(node.operation == "ReduceSum" for node in self.graph.nodes):
                return 0
            warps = self.schedule.threads_per_row // 32
            bytes_required = 4 * self.schedule.rows_per_block * warps if warps > 1 else 0
            return bytes_required
        if isinstance(self.schedule, TransposeSchedule):
            return 4096
        return 0
