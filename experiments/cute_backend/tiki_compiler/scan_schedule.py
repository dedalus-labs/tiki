"""Tile geometry and emitted-kernel metadata for associative scans."""

from dataclasses import dataclass

from .graph import Shape
from .schedule import Schedule, UnsupportedScheduleError


@dataclass(frozen=True)
class ScanSchedule:
    arch: str = "sm_90"
    threads: int = 128
    elements_per_thread: int = 4

    def __post_init__(self) -> None:
        Schedule(arch=self.arch, threads=self.threads)
        if type(self.elements_per_thread) is not int or not (1 <= self.elements_per_thread <= 16):
            raise UnsupportedScheduleError("elements_per_thread must be 1 to 16")

    @property
    def tile(self) -> int:
        tile = self.threads * self.elements_per_thread
        return tile

    @property
    def warps(self) -> int:
        warps = self.threads // 32
        return warps


@dataclass(frozen=True, kw_only=True)
class ScanLowered:
    schedule: ScanSchedule
    name: str
    mlir: str
    grid: tuple[int, int, int]
    output_shapes: tuple[Shape, ...]
    shared_memory_bytes: int
