"""One operation vocabulary for capture, scalar lowering, and frozen-graph replay."""

from collections.abc import Callable
from enum import Enum
from functools import partial

import tiki as tk


class UnsupportedGraphError(ValueError):
    """The exported graph is outside the compiler's supported operation subset."""


class Operation(Enum):
    # Names match Tiki's normalized export names, so consumers need no name table.
    Add = (2, tk.add, "arith.addf {0}, {1} : f32")
    Subtract = (2, tk.subtract, "arith.subf {0}, {1} : f32")
    Multiply = (2, tk.multiply, "arith.mulf {0}, {1} : f32")
    Negative = (1, tk.negative, "arith.negf {0} : f32")
    Square = (1, tk.square, "arith.mulf {0}, {0} : f32")
    Rsqrt = (1, tk.rsqrt, "math.rsqrt {0} : f32")
    Broadcast = (1, tk.broadcast_to, "{0}", True)
    ReduceSum = (1, partial(tk.sum, axis=-1, keepdims=True), None)
    Transpose = (1, partial(tk.transpose, axes=(1, 0)), None)

    def __init__(
        self,
        arity: int,
        evaluate: Callable[..., tk.array],
        scalar_template: str | None,
        takes_shape: bool = False,
    ) -> None:
        self.arity = arity
        self.evaluate = evaluate
        self.scalar_template = scalar_template
        self.takes_shape = takes_shape

    @classmethod
    def require(cls, name: str) -> "Operation":
        try:
            return cls[name]
        except KeyError as error:
            raise UnsupportedGraphError(f"unsupported Tiki primitive: {name}") from error

    def replay(self, inputs: tuple[tk.array, ...], shape: tuple[int, ...]) -> tk.array:
        if self.takes_shape:
            return self.evaluate(*inputs, shape)
        return self.evaluate(*inputs)

    def expression(self, inputs: tuple[str, ...]) -> str:
        if self.scalar_template is None:
            raise UnsupportedGraphError(f"{self.name} requires a cooperative schedule")
        return self.scalar_template.format(*inputs)
