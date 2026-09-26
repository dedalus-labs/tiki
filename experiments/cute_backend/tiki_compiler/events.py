"""Describe the native export callback ABI; decode it only in capture.py.

Each header has mandatory fields. Only collective primitives carry group data.
The C++ callback owns these records; they are not a user-provided interchange format.
"""

from typing import Literal, NotRequired, TypedDict

from mlx import core

from .graph import Shape

type Descriptor = tuple[str, Shape, core.Dtype]
type PrimitiveArgument = bool | int | list[int] | tuple[int, ...]


class InputsEvent(TypedDict):
    type: Literal["inputs"]
    inputs: list[Descriptor]


class OutputsEvent(TypedDict):
    type: Literal["outputs"]
    outputs: list[Descriptor]


class ConstantsEvent(TypedDict):
    type: Literal["constants"]
    constants: list[tuple[str, core.array]]


class KeywordsEvent(TypedDict):
    type: Literal["keyword_inputs"]
    keywords: list[tuple[str, str]]


class PrimitiveEvent(TypedDict):
    type: Literal["primitive"]
    name: str
    inputs: list[Descriptor]
    outputs: list[Descriptor]
    arguments: list[PrimitiveArgument]
    stream: core.Stream
    group: NotRequired[core.distributed.Group]
    group_index: NotRequired[int]


type ExportEvent = InputsEvent | OutputsEvent | ConstantsEvent | KeywordsEvent | PrimitiveEvent
