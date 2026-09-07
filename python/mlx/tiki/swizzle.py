# Copyright © 2026 Dedalus Labs, Inc.

"""Rust-owned XOR transforms shared by layouts and compiler schedules."""

from operator import index
from typing import SupportsIndex

from mlx.tiki._layout import LayoutError
from mlx.tiki._layout import Swizzle as NativeSwizzle


def _integer(value: SupportsIndex, field: str) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError("boolean index value")
        integral = index(value)
    except TypeError as error:
        raise LayoutError(
            f"swizzle {field} must be an integer, got {value!r}"
        ) from error
    if not -(2**63) <= integral < 2**63:
        raise LayoutError(
            f"swizzle {field} must fit a signed 64-bit integer, got {integral}"
        )
    return integral


class Swizzle(NativeSwizzle):
    """Permute nonnegative element offsets with disjoint XOR bit fields.

    ``bits`` is each field's width. ``base`` is the number of low bits below
    both fields. Positive ``shift`` copies high bits toward low bits. Negative
    ``shift`` copies low bits toward high bits. Parameters are immutable.

    Rust validates ``abs(shift) >= bits`` and requires both fields to fit bits
    0 through 62. The same transform is its own inverse. Compose it with a
    layout to supply the coordinate domain, without creating array storage.
    """

    __slots__ = ()

    def __init__(self, bits: int, base: int, shift: int) -> None:
        super().__init__(
            _integer(bits, "bits"), _integer(base, "base"), _integer(shift, "shift")
        )

    def __call__(self, offset: SupportsIndex) -> int:
        """Transform an integral element offset without dereferencing it."""
        return super().__call__(_integer(offset, "index"))

    def __str__(self) -> str:
        return f"SW_{self.bits}_{self.base}_{self.shift}"

    def __repr__(self) -> str:
        return f"Swizzle({self.bits}, {self.base}, {self.shift})"
