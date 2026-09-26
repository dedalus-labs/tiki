"""Compile float32 MLX functions into explicitly scheduled CuTe/native programs."""

from collections.abc import Callable

from tiki_compiler.artifact import CudaIo
from tiki_compiler.compiled import Compiled
from tiki_compiler.execution import BackendUnavailableError as BackendUnavailableError
from tiki_compiler.graph import ArrayFunction
from tiki_compiler.graph import UnsupportedGraphError as UnsupportedGraphError
from tiki_compiler.schedule import (
    RowSchedule as RowSchedule,
)
from tiki_compiler.schedule import (
    Schedule as Schedule,
)
from tiki_compiler.schedule import (
    Swizzle as Swizzle,
)
from tiki_compiler.schedule import (
    TransposeSchedule as TransposeSchedule,
)
from tiki_compiler.schedule import (
    UnsupportedScheduleError as UnsupportedScheduleError,
)

DEFAULT_SCHEDULE = Schedule()


def compile(
    *,
    backend: str = "cute",
    schedule: Schedule | RowSchedule | TransposeSchedule = DEFAULT_SCHEDULE,
) -> Callable[[ArrayFunction], Compiled]:
    """Specialize a pure array function; captured Python values are frozen per shape."""
    if backend != "cute":
        raise UnsupportedScheduleError(f"unsupported backend: {backend}")

    io = CudaIo()

    def decorate(function: ArrayFunction) -> Compiled:
        compiled = Compiled(io=io, function=function, schedule=schedule)
        return compiled

    return decorate
