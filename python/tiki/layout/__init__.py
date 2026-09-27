# Copyright © 2026 Dedalus Labs, Inc.

"""Tiki: CuTe-native layouts for Tiki arrays.

A Tiki array follows the zop tensor model: an Engine paired with a first-class
CuTe ``Layout``, so ``tensor[coordinate] == engine[layout(coordinate)]``. The
layout algebra and the validated Swizzle index transform are tiki-cute, Tiki's
Rust implementation of CuTe, reached through the ``tiki.layout._cute`` extension.
Generic composition preserves the exact ``outer o {offset} o inner`` map and
its slicing invariant.
"""

try:
    from tiki.layout import _cute  # noqa: F401
except ImportError as error:
    raise ImportError(
        "tiki.layout needs its Rust extension tiki.layout._cute, which a source build "
        "of Tiki compiles from tiki/cute. No Python fallback exists"
    ) from error

from tiki.layout.composed import ComposedLayout, check_swizzle, slice_and_offset
from tiki.layout.engine import ArrayEngine
from tiki.layout.layout import (
    F2,
    Array,
    E,
    Engine,
    ImplicitAccessor,
    Layout,
    LayoutError,
    MutableEngine,
    Ptr,
    Swizzle,
    Tensor,
    TransformAccessor,
    basis,
    blocked_product,
    coalesce,
    compatible,
    complement,
    compose,
    congruent,
    coshape,
    cosize,
    crd2idx,
    depth,
    flatten,
    identity_tensor,
    idx2crd,
    is_layout,
    left_inverse,
    logical_divide,
    logical_product,
    make_layout,
    make_tensor,
    nullspace,
    raked_product,
    rank,
    recast,
    right_inverse,
    size,
    unflatten,
    zipped_divide,
)
from tiki.layout.tensor import expand, from_array, realize, squeeze, unsqueeze

__all__ = [
    "Array",
    "ArrayEngine",
    "ComposedLayout",
    "E",
    "Engine",
    "F2",
    "ImplicitAccessor",
    "Layout",
    "LayoutError",
    "MutableEngine",
    "Ptr",
    "Swizzle",
    "Tensor",
    "TransformAccessor",
    "basis",
    "check_swizzle",
    "blocked_product",
    "coalesce",
    "compatible",
    "complement",
    "compose",
    "congruent",
    "coshape",
    "cosize",
    "crd2idx",
    "depth",
    "flatten",
    "from_array",
    "realize",
    "expand",
    "squeeze",
    "unsqueeze",
    "identity_tensor",
    "idx2crd",
    "is_layout",
    "left_inverse",
    "logical_divide",
    "logical_product",
    "make_layout",
    "make_tensor",
    "nullspace",
    "raked_product",
    "rank",
    "recast",
    "right_inverse",
    "size",
    "slice_and_offset",
    "unflatten",
    "zipped_divide",
]
