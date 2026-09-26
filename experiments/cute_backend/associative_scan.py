"""``associative_scan`` on Tiki's tiled scan kernels, with registered derivatives.

The interface is ``jax.lax.associative_scan``: ``fn`` combines two pytrees of
leaves and must be associative; element ``t`` of the result is ``fn`` folded
over elements ``0..t``, or ``t..length-1`` with ``reverse``. Forward evaluation is
the tile kernel, a recursive scan of the tile aggregates, and the apply
kernel, all specialized on the live layout of every leaf.

The derivatives use the sequential reading of the scan, ``y_t = fn(y_{t-1},
x_t)``, which equals the tree because ``fn`` is associative. With ``J_y(t)``
and ``J_x(t)`` the Jacobians of ``fn`` at step ``t`` (``size x size`` per position
for ``size`` leaves), the cotangent obeys the reverse affine recurrence
``gy_t = g_t + J_y(t+1)^T gy_{t+1}`` and the tangent the forward one
``dy_t = J_y(t) dy_{t-1} + J_x(t) dx_t``. Both are associative scans over
``(matrix, vector)`` pairs, so each derivative is the same kernel family with
the matrix-affine combine, plus two elementwise kernels for the Jacobians and
the matrix-vector products. This is how JAX derives its native ``cumsum``
gradient, generalized from ``J = 1`` to the combine's Jacobians.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import cast, overload

from mlx import core
from mlx.utils import tree_flatten, tree_unflatten

from tiki_compiler.artifact import CudaIo
from tiki_compiler.scan_runtime import ScanContractError, ScanOp
from tiki_compiler.scan_schedule import ScanSchedule as ScanSchedule

# MLX's tree paths normalize tuples to lists when rebuilding a tree.
type ArrayTree = core.array | Sequence[ArrayTree] | Mapping[str, ArrayTree]
# The two operands' leaf structure is checked against the captured paths.
type TreeCombine = Callable[..., ArrayTree]
DEFAULT_SCAN_SCHEDULE = ScanSchedule()


def flatten(tree: ArrayTree) -> list[tuple[str, core.array]]:
    leaves: list[tuple[str, core.array]] = []
    tree_flatten(tree, destination=leaves)
    if any(not isinstance(leaf, core.array) for _, leaf in leaves):
        raise ScanContractError("every pytree leaf must be an MLX array")
    return leaves


def unflatten(paths: tuple[str, ...], leaves: tuple[core.array, ...]) -> ArrayTree:
    # The native utility erases its result type; these paths contain only array leaves.
    result = cast(ArrayTree, tree_unflatten(list(zip(paths, leaves, strict=True))))
    return result


@lru_cache(maxsize=32)
def operation(
    fn: TreeCombine,
    paths: tuple[str, ...],
    axis: int,
    schedule: ScanSchedule,
) -> ScanOp:
    size = len(paths)

    def combine(*args: core.array) -> tuple[core.array, ...]:
        left = unflatten(paths, args[:size])
        right = unflatten(paths, args[size:])
        result = flatten(fn(left, right))
        if tuple(path for path, _ in result) != paths:
            raise ScanContractError("fn must return the structure of elems")
        result = tuple(leaf for _, leaf in result)
        return result

    result = ScanOp(io=CudaIo(), combine=combine, leaves=size, axis=axis, schedule=schedule)
    return result


@overload
def associative_scan(
    fn: Callable[[core.array, core.array], core.array],
    elems: core.array,
    *,
    reverse: bool = False,
    axis: int = 0,
    schedule: ScanSchedule = DEFAULT_SCAN_SCHEDULE,
) -> core.array: ...


@overload
def associative_scan(
    fn: TreeCombine,
    elems: ArrayTree,
    *,
    reverse: bool = False,
    axis: int = 0,
    schedule: ScanSchedule = DEFAULT_SCAN_SCHEDULE,
) -> ArrayTree: ...


def associative_scan(
    fn: TreeCombine,
    elems: ArrayTree,
    *,
    reverse: bool = False,
    axis: int = 0,
    schedule: ScanSchedule = DEFAULT_SCAN_SCHEDULE,
) -> ArrayTree:
    """Scan ``elems`` along ``axis`` with the associative operation ``fn``."""
    flat = flatten(elems)
    if not flat:
        raise ScanContractError("elems must contain at least one array")
    paths = tuple(path for path, _ in flat)
    leaves = tuple(leaf for _, leaf in flat)
    ndim = leaves[0].ndim
    if not -ndim <= axis < ndim:
        raise ScanContractError(f"axis {axis} is out of range for {ndim} dimensions")
    op = operation(fn, paths, axis % ndim, schedule)
    results = op.reverse(*leaves) if reverse else op(*leaves)
    result = unflatten(paths, results)
    return result
