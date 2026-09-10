"""Select a supported lowering from a function, schedule, layout, and stream."""

from functools import lru_cache
from typing import assert_never

from mlx import core

from .capture import TensorOperation, from_events, trace
from .elementwise import lower
from .execution import packs_views
from .graph import ArrayFunction, Profile, dense_strides
from .lowered import Lowered
from .partition import partition
from .program import Program
from .row_reduction import lower_row
from .schedule import RowSchedule, Schedule, TransposeSchedule, UnsupportedScheduleError
from .transpose import lower_transpose


@lru_cache(maxsize=32)
def specialize(
    function: ArrayFunction,
    schedule: Schedule | RowSchedule | TransposeSchedule,
    profiles: tuple[Profile, ...],
    stream: core.Stream,
) -> Lowered | Program:
    if packs_views(schedule):
        profiles = tuple(
            Profile(shape=shape, strides=dense_strides(shape)) for shape, _ in profiles
        )
    with core.stream(stream):
        captured = from_events(trace(function, profiles), profiles)
    if any(
        not isinstance(operation, TensorOperation) or operation.stream != stream
        for operation in captured.operations
    ):
        if not isinstance(schedule, Schedule):
            raise UnsupportedScheduleError(
                "native operations and stream boundaries require an elementwise Schedule"
            )
        result = partition(captured, schedule)
        return result
    graph = captured.tensor_graph()
    match schedule:
        case RowSchedule():
            lowered = lower_row(graph, schedule)
        case TransposeSchedule():
            lowered = lower_transpose(graph, schedule)
        case Schedule():
            lowered = lower(graph, schedule)
        case _:
            assert_never(schedule)
    return lowered
