"""Execute a fixed sequence of CuTe regions and native MLX graph operations."""

from dataclasses import dataclass
from typing import Protocol, assert_never

from mlx import core

from .graph import Arrays, Scalar, Shape, Value
from .lowered import Lowered
from .native import Collective, GroupIndex, Matmul


class KernelLauncher(Protocol):
    def __call__(self, lowered: Lowered, inputs: Arrays, stream: core.Stream) -> Arrays: ...


@dataclass(frozen=True, kw_only=True)
class Kernel:
    lowered: Lowered
    stream: core.Stream

    @property
    def operation(self) -> str:
        return "CuTe"


@dataclass(frozen=True, kw_only=True)
class Program:
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    constants: tuple[Scalar, ...]
    stages: tuple[Matmul | Collective | Kernel, ...]

    @property
    def output_shapes(self) -> tuple[Shape, ...]:
        shapes = tuple(value.shape for value in self.outputs)
        return shapes

    def launch(self, inputs: Arrays, launch_kernel: KernelLauncher) -> Arrays:
        """Build a lazy graph; serialize collectives only within their communicator."""
        values = dict(zip((value.name for value in self.inputs), inputs, strict=True))
        values.update(
            (name, core.array(value, dtype=core.float32)) for name, value in self.constants
        )
        completions: dict[GroupIndex, core.array] = {}
        for stage in self.stages:
            match stage:
                case Kernel():
                    region = stage.lowered.graph
                    arguments = tuple(values[value.name] for value in region.inputs)
                    outputs = launch_kernel(stage.lowered, arguments, stage.stream)
                    values.update(
                        zip((value.name for value in region.outputs), outputs, strict=True)
                    )
                case Matmul():
                    values[stage.output.name] = core.matmul(
                        values[stage.left], values[stage.right], stream=stage.stream
                    )
                case Collective():
                    output = stage.launch(
                        values[stage.input], after=completions.get(stage.group_index)
                    )
                    values[stage.output.name] = output
                    completions[stage.group_index] = output
                case _:
                    assert_never(stage)
        result = tuple(values[value.name] for value in self.outputs)
        return result
