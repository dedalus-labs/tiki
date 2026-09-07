# Copyright © 2026 Dedalus Labs, Inc.

"""Layouts: hierarchical ``Shape:Stride`` coordinate maps from the vendored PyCuTe.

Integer-stride operations use the pinned PyCuTe algebra. ``compose`` also
retains nonlinear transforms and their internal offsets. ``basis`` is ``E``,
and ``cosize`` is the size of ``coshape``, not a storage-bounds proof.
"""

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar, cast

from mlx.tiki._layout import LayoutError
from mlx.tiki._pycute import (
    F2,
    Accessor,
    Array,
    E,
    ImplicitAccessor,
)
from mlx.tiki._pycute import Layout as ReferenceLayout
from mlx.tiki._pycute import (
    LayoutBase,
    MutableAccessor,
    Ptr,
    Shape,
    Tensor,
    TransformAccessor,
)
from mlx.tiki._pycute import blocked_product as _blocked_product
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
from mlx.tiki._pycute import make_layout as _make_layout
from mlx.tiki._pycute import (
    make_tensor,
)
from mlx.tiki._pycute import nullspace as _nullspace
from mlx.tiki._pycute import raked_product as _raked_product
from mlx.tiki._pycute import (
    rank,
)
from mlx.tiki._pycute import recast as _recast
from mlx.tiki._pycute import right_inverse as _right_inverse
from mlx.tiki._pycute import (
    size,
    unflatten,
)
from mlx.tiki._pycute import zipped_divide as _zipped_divide
from mlx.tiki._pycute.layout import Tiler
from mlx.tiki.affine import Layout
from mlx.tiki.composed import ComposedLayout
from mlx.tiki.swizzle import Swizzle

Engine = Accessor
MutableEngine = MutableAccessor
basis = E


Parameters = ParamSpec("Parameters")
Return = TypeVar("Return")


def _typed(operation: Callable[Parameters, Return]) -> Callable[Parameters, Return]:
    """Re-raise PyCuTe's precondition failures as ``LayoutError``."""

    @wraps(operation)
    def guarded(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Return:
        if any(
            isinstance(value, ComposedLayout)
            or isinstance(value, Tensor)
            and isinstance(value.layout, ComposedLayout)
            for value in args
        ):
            raise LayoutError(
                f"{operation.__name__} requires a stride layout. Transform the domain before composing"
            )
        try:
            result = operation(*args, **kwargs)
        except (ValueError, TypeError) as error:
            raise LayoutError(str(error)) from error
        if type(result) is ReferenceLayout:
            return cast(Return, Layout._set(result.shape, result.stride))
        if type(result) is Tensor and type(result.layout) is ReferenceLayout:
            layout = Layout._set(result.layout.shape, result.layout.stride)
            return cast(Return, Tensor(result.accessor, layout))
        return result

    return guarded


coalesce = _typed(_coalesce)
_affine_compose = _typed(_composition)
make_layout = _typed(_make_layout)
blocked_product = _typed(_blocked_product)
raked_product = _typed(_raked_product)
complement = _typed(_complement)
logical_divide = _typed(_logical_divide)
zipped_divide = _typed(_zipped_divide)
logical_product = _typed(_logical_product)
right_inverse = _typed(_right_inverse)
left_inverse = _typed(_left_inverse)
nullspace = _typed(_nullspace)
recast = _typed(_recast)


def compose(
    outer: LayoutBase | Tensor | Swizzle | Tiler,
    inner: LayoutBase | Tiler,
    *,
    offset: int = 0,
    mode: int | tuple[int, ...] = (),
) -> LayoutBase | Tensor:
    """Compose coordinate maps, retaining offsets inside nonlinear transforms.

    Integer-affine composition uses PyCuTe's algebra. A nonlinear operand or
    an explicit offset stays an inspectable expression over the inner domain.
    """
    if type(offset) is not int:
        raise LayoutError("composition offset must be an integer")
    if (
        isinstance(outer, (Swizzle, ComposedLayout))
        or isinstance(inner, ComposedLayout)
        or offset != 0
    ):
        if mode != ():
            raise LayoutError(
                "compose the selected domain before applying a nonlinear transform"
            )
        if not isinstance(outer, (Swizzle, ReferenceLayout, ComposedLayout)):
            raise LayoutError("composition outer must be a Swizzle or a layout")
        if not isinstance(inner, (ReferenceLayout, ComposedLayout)):
            raise LayoutError("composition inner must supply a layout domain")
        return ComposedLayout(outer, offset, inner)
    return _affine_compose(outer, inner, mode=mode)


def coshape(layout: Layout) -> Shape:
    """Shape of the codomain: one past the largest offset in each codomain mode."""
    if not isinstance(layout, ReferenceLayout):
        raise LayoutError("coshape requires a stride layout")
    return layout._coshape()


def cosize(layout: Layout) -> int:
    """Size of the codomain. This bounds offsets; it is not a storage-bounds proof."""
    return size(coshape(layout))
