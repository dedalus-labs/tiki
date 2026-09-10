"""Normalize MLX callback results at the array/tuple boundary."""

from mlx import core

from .graph import ArrayResult, Profile, UnsupportedGraphError


def arrays(value: core.array | tuple[core.array, ...]) -> tuple[core.array, ...]:
    """MLX passes a bare array to derivative callbacks of single-input functions."""
    result = (value,) if isinstance(value, core.array) else tuple(value)
    return result


def single(value: ArrayResult) -> core.array:
    """Require one result for a derivative kernel that produces one cotangent."""
    values = arrays(value)
    if len(values) != 1:
        raise UnsupportedGraphError(f"expected one array result, got {len(values)}")
    result = values[0]
    return result


def profile(array: core.array) -> Profile:
    """Record the evaluated array's layout as an immutable specialization key."""
    result = Profile(shape=tuple(array.shape), strides=tuple(array.strides))
    return result
