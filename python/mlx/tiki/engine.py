# Copyright © 2026 Dedalus Labs, Inc.

"""MLX arrays as Engines, the storage half of a Tiki tensor.

An ``ArrayEngine`` is a flat MLX buffer plus an element offset. Element reads
and writes go through MLX indexing, which is exact but slow; that is what the
reference algebra expects of an Accessor. A whole layout is realized as one
zero-copy view by ``mlx.tiki.tensor.realize``.

MLX does not expose the strides of a lazy view in Python, so an engine is
always built over a flattened base and the ``Layout`` is authoritative.
"""

from typing import Any

import mlx.core as mx
from mlx.tiki._pycute import MutableAccessor


class ArrayEngine(MutableAccessor):
    def __init__(self, base: mx.array, offset: int = 0):
        if base.ndim != 1:
            raise ValueError(
                f"ArrayEngine needs a flat base array, got shape {base.shape}"
            )
        self.base = base
        self.offset = int(offset)

    def __add__(self, delta: Any) -> "ArrayEngine":
        return ArrayEngine(self.base, self.offset + int(delta))

    def __getitem__(self, index: Any) -> Any:
        return self.base[self.offset + int(index)].item()

    def __setitem__(self, index: Any, value: Any) -> None:
        self.base[self.offset + int(index)] = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayEngine):
            return NotImplemented
        return self.base is other.base and self.offset == other.offset

    def __repr__(self) -> str:
        return f"ArrayEngine({self.base.dtype}[{self.base.size}] + {self.offset})"
