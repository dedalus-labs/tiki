"""Native stages retain MLX operation identity, stream, and communicator ownership."""

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, assert_never

from mlx import core

from .graph import Symbol, Value

GroupIndex = NewType("GroupIndex", int)


class CollectiveOperation(StrEnum):
    SUM = "AllSum"
    MIN = "AllMin"
    MAX = "AllMax"
    GATHER = "AllGather"
    SUM_SCATTER = "SumScatter"


@dataclass(frozen=True, kw_only=True)
class Matmul:
    left: Symbol
    right: Symbol
    output: Value
    stream: core.Stream

    @property
    def operation(self) -> str:
        return "Matmul"

    @property
    def inputs(self) -> tuple[Symbol, Symbol]:
        inputs = (self.left, self.right)
        return inputs


@dataclass(frozen=True, kw_only=True)
class Collective:
    operation: CollectiveOperation
    input: Symbol
    output: Value
    stream: core.Stream
    group: core.distributed.Group
    group_index: GroupIndex

    @property
    def inputs(self) -> tuple[Symbol]:
        inputs = (self.input,)
        return inputs

    def launch(self, value: core.array, after: core.array | None) -> core.array:
        if after is not None:
            with core.stream(self.stream):
                value = core.depends(value, after)
        match self.operation:
            case CollectiveOperation.SUM:
                output = core.distributed.all_sum(value, group=self.group, stream=self.stream)
            case CollectiveOperation.MIN:
                output = core.distributed.all_min(value, group=self.group, stream=self.stream)
            case CollectiveOperation.MAX:
                output = core.distributed.all_max(value, group=self.group, stream=self.stream)
            case CollectiveOperation.GATHER:
                output = core.distributed.all_gather(value, group=self.group, stream=self.stream)
            case CollectiveOperation.SUM_SCATTER:
                output = core.distributed.sum_scatter(value, group=self.group, stream=self.stream)
            case _:
                assert_never(self.operation)
        return output
