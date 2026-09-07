# Copyright © 2026 Dedalus Labs, Inc.

"""MLX arrays as Engines, the storage half of a Tiki tensor.

An ``ArrayEngine`` is a flat MLX buffer plus an element offset. Element reads
and writes go through MLX indexing, which is exact but slow; that is what the
reference algebra expects of an Accessor. A whole layout is realized as one
zero-copy view by ``mlx.tiki.tensor.realize``.

An Engine uses flat storage. ``from_array`` normalizes the base explicitly
so its layout is authoritative when constructing an ``as_strided`` view.
"""

from operator import index
from typing import SupportsIndex

import mlx.core as mx
from mlx.tiki._pycute import MutableAccessor
from mlx.tiki.layout import LayoutError


def integer_offset(value: SupportsIndex) -> int:
    """Require an integral Engine displacement without coercing another algebra."""
    try:
        if isinstance(value, bool):
            raise TypeError("boolean offset")
        return index(value)
    except TypeError as error:
        raise LayoutError(f"Engine offsets must be integers, got {value!r}") from error


class ArrayEngine(MutableAccessor):
    def __init__(self, base: mx.array, offset: int = 0) -> None:
        if not isinstance(base, mx.array):
            raise LayoutError("ArrayEngine needs an MLX array")
        if base.ndim != 1:
            raise LayoutError(
                f"ArrayEngine needs a flat base array, got shape {base.shape}"
            )
        self.base = base
        self.offset = integer_offset(offset)
        if not 0 <= self.offset <= base.size:
            raise LayoutError(
                f"Engine offset {self.offset} is outside [0, {base.size}]"
            )

    def __add__(self, delta: SupportsIndex) -> "ArrayEngine":
        return ArrayEngine(self.base, self.offset + integer_offset(delta))

    def _position(self, value: SupportsIndex) -> int:
        position = self.offset + integer_offset(value)
        if not 0 <= position < self.base.size:
            raise LayoutError(
                f"Engine address {position} is outside [0, {self.base.size})"
            )
        return position

    def __getitem__(self, index: SupportsIndex) -> bool | int | float | complex:
        return self.base[self._position(index)].item()

    def __setitem__(
        self, index: SupportsIndex, value: bool | int | float | complex
    ) -> None:
        self.base[self._position(index)] = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayEngine):
            return NotImplemented
        return self.base is other.base and self.offset == other.offset

    def __repr__(self) -> str:
        return f"ArrayEngine({self.base.dtype}[{self.base.size}] + {self.offset})"
