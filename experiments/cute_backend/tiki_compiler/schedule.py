"""Validated launch schedules for the supported CUDA kernels."""

from dataclasses import dataclass


class UnsupportedScheduleError(ValueError):
    """The requested schedule is outside the supported CUDA target."""


@dataclass(frozen=True)
class Schedule:
    arch: str = "sm_90"
    threads: int = 128
    elements_per_thread: int = 1

    def __post_init__(self) -> None:
        if self.arch not in ("sm_89", "sm_90"):
            raise UnsupportedScheduleError("only sm_89 and sm_90 are supported")
        if type(self.threads) is not int or type(self.elements_per_thread) is not int:
            raise UnsupportedScheduleError("schedule dimensions must be integers")
        if self.threads not in (32, 64, 128, 256):
            raise UnsupportedScheduleError("threads must be 32, 64, 128, or 256")
        if self.elements_per_thread not in (1, 2, 4):
            raise UnsupportedScheduleError("elements_per_thread must be 1, 2, or 4")


@dataclass(frozen=True)
class RowSchedule:
    arch: str = "sm_90"
    threads_per_row: int = 128
    rows_per_block: int = 1

    def __post_init__(self) -> None:
        if type(self.rows_per_block) is not int or self.rows_per_block not in (
            1,
            2,
            4,
            8,
            16,
            32,
        ):
            raise UnsupportedScheduleError("rows_per_block must be a power of two from 1 to 32")
        if type(self.threads_per_row) is not int or self.threads_per_row not in (
            8,
            16,
            32,
            64,
            128,
            256,
        ):
            raise UnsupportedScheduleError("threads_per_row must be 8, 16, 32, 64, 128, or 256")
        Schedule(arch=self.arch, threads=self.threads)

    @property
    def threads(self) -> int:
        threads = self.threads_per_row * self.rows_per_block
        return threads


@dataclass(frozen=True)
class Swizzle:
    bits: int = 5
    base: int = 0
    shift: int = 5

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.bits, self.base, self.shift)):
            raise UnsupportedScheduleError("swizzle parameters must be integers")
        if not (0 <= self.bits <= 5 and 0 <= self.base <= 5 - self.bits and self.shift == 5):
            raise UnsupportedScheduleError(
                "32x32 transpose requires 0 <= bits + base <= 5 and shift=5"
            )

    def offset(self, index: int) -> int:
        mask = ((1 << self.bits) - 1) << self.base
        offset = index ^ ((index >> self.shift) & mask)
        return offset


@dataclass(frozen=True)
class TransposeSchedule:
    arch: str = "sm_90"
    threads: int = 128
    swizzle: Swizzle = Swizzle()

    def __post_init__(self) -> None:
        Schedule(arch=self.arch, threads=self.threads)
