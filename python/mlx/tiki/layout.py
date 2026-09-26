# Copyright © 2026 Dedalus Labs, Inc.

"""Layouts: hierarchical ``Shape:Stride`` coordinate maps from the vendored PyCuTe.

Names follow zop, which follows CuTe. ``compose`` is PyCuTe's ``composition``,
``basis`` is ``E``, and ``cosize`` is the size of ``coshape``. Every algebra
operation raises ``LayoutError`` when a precondition (congruence, divisibility,
invertibility) fails; nothing degrades to a weaker layout.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

from mlx.tiki._pycute import (
    F2,
    Accessor,
    Array,
    E,
    ImplicitAccessor,
    Layout,
    MutableAccessor,
    Ptr,
)
from mlx.tiki._pycute import Swizzle as _Swizzle
from mlx.tiki._pycute import (
    Tensor,
    TransformAccessor,
    blocked_product,
)
from mlx.tiki._pycute import coalesce as _coalesce
from mlx.tiki._pycute import (
    compatible,
)
from mlx.tiki._pycute import complement as _complement
from mlx.tiki._pycute import composition as _composition
from mlx.tiki._pycute import (
    congruent,
    crd2idx,
    depth,
    flatten,
    identity_tensor,
    idx2crd,
    is_layout,
)
from mlx.tiki._pycute import left_inverse as _left_inverse
from mlx.tiki._pycute import logical_divide as _logical_divide
from mlx.tiki._pycute import logical_product as _logical_product
from mlx.tiki._pycute import (
    make_layout,
    make_tensor,
)
from mlx.tiki._pycute import nullspace as _nullspace
from mlx.tiki._pycute import (
    raked_product,
    rank,
)
from mlx.tiki._pycute import recast as _recast
from mlx.tiki._pycute import right_inverse as _right_inverse
from mlx.tiki._pycute import (
    size,
    unflatten,
)
from mlx.tiki._pycute import zipped_divide as _zipped_divide

Engine = Accessor
MutableEngine = MutableAccessor
basis = E


class LayoutError(ValueError):
    """A layout precondition failed: congruence, divisibility, or invertibility."""


class Swizzle(_Swizzle):
    """A CuTe bit permutation whose source and destination fields are disjoint."""

    def __init__(self, bits: int, base: int, shift: int) -> None:
        if any(type(value) is not int for value in (bits, base, shift)):
            raise LayoutError("swizzle bits, base, and shift must be integers")
        if bits < 0 or base < 0 or abs(shift) < bits:
            raise LayoutError(
                f"invalid swizzle ({bits}, {base}, {shift}). Require bits >= 0, base >= 0, and abs(shift) >= bits"
            )
        super().__init__(bits, base, shift)


def _typed(operation: Callable[..., Any]) -> Callable[..., Any]:
    """Re-raise PyCuTe's precondition failures as ``LayoutError``."""

    @wraps(operation)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except ValueError as error:
            raise LayoutError(str(error)) from error

    return guarded


coalesce = _typed(_coalesce)
compose = _typed(_composition)
complement = _typed(_complement)
logical_divide = _typed(_logical_divide)
zipped_divide = _typed(_zipped_divide)
logical_product = _typed(_logical_product)
right_inverse = _typed(_right_inverse)
left_inverse = _typed(_left_inverse)
nullspace = _typed(_nullspace)
recast = _typed(_recast)


def coshape(layout: Layout) -> Any:
    """Shape of the codomain: one past the largest offset in each codomain mode."""
    return layout._coshape()


def cosize(layout: Layout) -> int:
    """Size of the codomain. This bounds offsets; it is not a storage-bounds proof."""
    return size(coshape(layout))
