"""Launch CuTe artifacts and cache native MLX graph replay for mixed programs."""

from collections.abc import Callable
from functools import lru_cache
from math import prod

from mlx import core

from .artifact import CudaIo, binary
from .graph import Arrays
from .lowered import Lowered
from .program import Program
from .schedule import RowSchedule, Schedule, TransposeSchedule


class BackendUnavailableError(RuntimeError):
    """Execution requires the selected MLX CUDA device."""


def packs_views(schedule: Schedule | RowSchedule | TransposeSchedule) -> bool:
    """Cooperative schedules address dense storage only, so their views are
    packed to row-major before launch and they specialize on the dense
    profile. The elementwise schedule addresses each view in place."""
    packed = not isinstance(schedule, Schedule)
    return packed


def launch_kernel(
    io: CudaIo, lowered: Lowered, inputs: tuple[core.array, ...], stream: core.Stream
) -> tuple[core.array, ...]:
    shapes = lowered.output_shapes
    if prod(lowered.graph.shape) == 0:
        outputs = tuple(core.zeros(shape, dtype=core.float32, stream=stream) for shape in shapes)
        return outputs
    outputs = tuple(
        core.fast.precompiled_cuda_kernel(
            name="tiki_fused",
            compiled_source=binary(io, lowered).cubin,
            inputs=list(inputs),
            output_shapes=list(shapes),
            output_dtypes=[core.float32] * len(shapes),
            scalars=[],
            grid=lowered.grid,
            threadgroup=(lowered.schedule.threads, 1, 1),
            shared_memory=lowered.shared_memory_bytes,
            ensure_row_contiguous=packs_views(lowered.schedule),
            stream=stream,
        )
    )
    return outputs


@lru_cache(maxsize=32)
def program_executor(io: CudaIo, program: Program) -> Callable[..., tuple[core.array, ...]]:
    """Cache native graph replay after specializing the CuTe regions."""

    def launch(lowered: Lowered, inputs: Arrays, stream: core.Stream) -> Arrays:
        outputs = launch_kernel(io, lowered, inputs, stream)
        return outputs

    def replay(*inputs: core.array) -> Arrays:
        outputs = program.launch(inputs, launch)
        return outputs

    executor = core.compile(replay)
    return executor
