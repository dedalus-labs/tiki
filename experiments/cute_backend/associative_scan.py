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

from collections.abc import Callable
from functools import lru_cache, reduce
from math import prod
from operator import add
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from graph import Graph, Profile, UnsupportedGraphError, capture
from scan_lowering import ScanLowered, ScanSchedule, lower_apply, lower_tile_scan
from tiki import BackendUnavailableError, Compiled, Schedule, _arrays, binary, profile

Leaves = tuple[mx.array, ...]
FlatCombine = Callable[..., tuple[mx.array, ...]]
DEFAULT_SCAN_SCHEDULE = ScanSchedule()


class ScanContractError(ValueError):
    """The leaves are outside the scan contract."""


def view(
    array: mx.array, axis: int, start: int | None, stop: int | None, step: int = 1
) -> mx.array:
    index: list[slice] = [slice(None)] * array.ndim
    index[axis] = slice(start, stop, step)
    return array[tuple(index)]


def flip(array: mx.array, axis: int) -> mx.array:
    """A negatively strided view; the kernels consume it in place."""
    return view(array, axis, None, None, -1)


def basis(array: mx.array, value: float) -> mx.array:
    """A constant tangent the graph capture accepts: a float32 scalar broadcast."""
    return mx.broadcast_to(mx.array(value, dtype=mx.float32), array.shape)


def affine_combine(size: int) -> FlatCombine:
    """``(A_l, b_l) o (A_r, b_r) = (A_r A_l, A_r b_l + b_r)`` over ``size*size + size`` leaves."""

    def combine(*args: mx.array) -> tuple[mx.array, ...]:
        a_left, b_left = args[: size * size], args[size * size : size * size + size]
        a_right, b_right = (
            args[size * size + size : 2 * size * size + size],
            args[2 * size * size + size :],
        )
        matrix = [
            reduce(
                add,
                (
                    a_right[row * size + inner] * a_left[inner * size + column]
                    for inner in range(size)
                ),
            )
            for row in range(size)
            for column in range(size)
        ]
        vector = [
            reduce(
                add,
                (a_right[row * size + inner] * b_left[inner] for inner in range(size)),
            )
            + b_right[row]
            for row in range(size)
        ]
        return (*matrix, *vector)

    return combine


def matvec(size: int, transpose: bool) -> Callable[..., tuple[mx.array, ...]]:
    """``M v`` (or ``M^T v``) over ``size*size`` matrix leaves and ``size`` vector leaves."""

    def function(*args: mx.array) -> tuple[mx.array, ...]:
        matrix, vector = args[: size * size], args[size * size :]
        if transpose:
            return tuple(
                reduce(
                    add,
                    (matrix[row * size + column] * vector[row] for row in range(size)),
                )
                for column in range(size)
            )
        return tuple(
            reduce(
                add,
                (
                    matrix[row * size + column] * vector[column]
                    for column in range(size)
                ),
            )
            for row in range(size)
        )

    return function


@lru_cache(maxsize=64)
def tile_kernel(
    combine: Graph, profiles: tuple[Profile, ...], axis: int, schedule: ScanSchedule
) -> ScanLowered:
    return lower_tile_scan(combine, profiles, axis, schedule)


@lru_cache(maxsize=64)
def apply_kernel(
    combine: Graph,
    shape: tuple[int, ...],
    axis: int,
    tiles: int,
    schedule: ScanSchedule,
) -> ScanLowered:
    return lower_apply(combine, shape, axis, tiles, schedule)


def launch(lowered: ScanLowered, inputs: Leaves) -> Leaves:
    if not mx.cuda.is_available():
        raise BackendUnavailableError("associative_scan execution requires MLX CUDA")
    if mx.device_info(mx.gpu)["architecture"] != lowered.schedule.arch:
        raise BackendUnavailableError(f"schedule requires {lowered.schedule.arch}")
    return tuple(
        mx.fast.precompiled_cuda_kernel(
            name=lowered.name,
            compiled_source=binary(lowered).cubin,
            inputs=list(inputs),
            output_shapes=list(lowered.output_shapes),
            output_dtypes=[mx.float32] * len(lowered.output_shapes),
            scalars=[],
            grid=lowered.grid,
            threadgroup=(lowered.schedule.threads, 1, 1),
            shared_memory=lowered.shared_memory_bytes,
            ensure_row_contiguous=False,
            stream=mx.gpu,
        )
    )


class ScanOp:
    """An associative scan of ``leaves`` float32 arrays along ``axis``."""

    def __init__(
        self, combine: FlatCombine, leaves: int, axis: int, schedule: ScanSchedule
    ):
        self.combine = combine
        self.leaves = leaves
        self.axis = axis
        self.schedule = schedule
        self.graph = capture(combine, (((), ()),) * (2 * leaves))
        if len(self.graph.outputs) != leaves:
            raise UnsupportedGraphError("the combine must return one array per leaf")
        self._function = mx.custom_function(self.forward)
        self._function.vjp(self._vjp)
        self._function.jvp(self._jvp)
        self._aggregate: ScanOp | None = None
        self._affine: ScanOp | None = None
        self._jacobian = Compiled(self.jacobian, Schedule())
        self._matvec = Compiled(matvec(leaves, transpose=False), Schedule())
        self._matvec_transposed = Compiled(matvec(leaves, transpose=True), Schedule())

    def __call__(self, *leaves: mx.array) -> Leaves:
        return _arrays(self._function(*leaves))

    def reverse(self, *leaves: mx.array) -> Leaves:
        axis = self.axis
        return tuple(
            flip(result, axis)
            for result in self(*(flip(leaf, axis) for leaf in leaves))
        )

    def check(self, leaves: Leaves) -> tuple[int, ...]:
        if len(leaves) != self.leaves:
            raise ScanContractError(f"expected {self.leaves} leaves, got {len(leaves)}")
        shape = tuple(leaves[0].shape)
        if any(
            tuple(leaf.shape) != shape or leaf.dtype != mx.float32 for leaf in leaves
        ):
            raise ScanContractError("all leaves must be float32 arrays of one shape")
        if not 0 <= self.axis < len(shape):
            raise ScanContractError(
                f"axis {self.axis} is out of range for shape {shape}"
            )
        if prod(shape) >= 2**31 - 1024:
            raise ScanContractError("element count exceeds signed 32-bit indexing")
        return shape

    def forward(self, *leaves: mx.array) -> Leaves:
        shape = self.check(leaves)
        if prod(shape) == 0:
            return tuple(
                mx.zeros(shape, dtype=mx.float32, stream=mx.gpu) for _ in leaves
            )
        profiles = tuple(profile(leaf) for leaf in leaves)
        lowered = tile_kernel(self.graph, profiles, self.axis, self.schedule)
        outputs = launch(lowered, leaves)
        local, aggregates = outputs[: self.leaves], outputs[self.leaves :]
        tiles = lowered.output_shapes[-1][1]
        if tiles == 1:
            return local
        if self._aggregate is None:
            self._aggregate = ScanOp(self.combine, self.leaves, 1, self.schedule)
        carry = self._aggregate.forward(*aggregates)
        applied = apply_kernel(self.graph, shape, self.axis, tiles, self.schedule)
        return launch(applied, (*carry, *local))

    @property
    def affine(self) -> "ScanOp":
        """The derivative recurrences as a scan over ``(size x size matrix, size vector)`` pairs."""
        if self._affine is None:
            size = self.leaves
            self._affine = ScanOp(
                affine_combine(size), size * size + size, self.axis, self.schedule
            )
        return self._affine

    def jacobian(self, *args: mx.array) -> tuple[mx.array, ...]:
        """``J_y`` then ``J_x`` entries, row-major ``(output row, input column)``, per position."""
        size = self.leaves
        primals = list(args)
        columns = []
        for column in range(2 * size):
            tangents = [
                basis(primal, float(row == column))
                for row, primal in enumerate(primals)
            ]
            columns.append(
                mx.jvp(lambda *a: list(self.combine(*a)), primals, tangents)[1]
            )
        left = [columns[column][row] for row in range(size) for column in range(size)]
        right = [
            columns[size + column][row] for row in range(size) for column in range(size)
        ]
        return (*left, *right)

    def _vjp(
        self,
        primals: mx.array | Leaves,
        cotangents: mx.array | Leaves,
        outputs: mx.array | Leaves,
    ) -> Leaves:
        x, g, y = _arrays(primals), _arrays(cotangents), _arrays(outputs)
        size, axis = self.leaves, self.axis
        length = x[0].shape[axis]
        head = [view(leaf, axis, 0, length - 1) for leaf in y]
        tail = [view(leaf, axis, 1, length) for leaf in x]
        jacobians = _arrays(self._jacobian(*head, *tail))
        left, right = jacobians[: size * size], jacobians[size * size :]
        pad = mx.zeros_like(view(x[0], axis, 0, 1))
        matrices = [
            mx.concatenate([left[column * size + row], pad], axis=axis)
            for row in range(size)
            for column in range(size)
        ]
        gy = self.affine.reverse(*matrices, *g)[size * size :]
        gx_tail = _arrays(
            self._matvec_transposed(
                *right, *(view(leaf, axis, 1, length) for leaf in gy)
            )
        )
        return tuple(
            mx.concatenate([view(gy[column], axis, 0, 1), gx_tail[column]], axis=axis)
            for column in range(size)
        )

    def _jvp(self, primals: mx.array | Leaves, tangents: mx.array | Leaves) -> Leaves:
        x, dx = _arrays(primals), _arrays(tangents)
        size, axis = self.leaves, self.axis
        length = x[0].shape[axis]
        y = self.forward(*x)
        head = [view(leaf, axis, 0, length - 1) for leaf in y]
        tail = [view(leaf, axis, 1, length) for leaf in x]
        jacobians = _arrays(self._jacobian(*head, *tail))
        left, right = jacobians[: size * size], jacobians[size * size :]
        pad = mx.zeros_like(view(x[0], axis, 0, 1))
        matrices = [mx.concatenate([pad, entry], axis=axis) for entry in left]
        driven = _arrays(
            self._matvec(*right, *(view(leaf, axis, 1, length) for leaf in dx))
        )
        vectors = [
            mx.concatenate([view(dx[row], axis, 0, 1), driven[row]], axis=axis)
            for row in range(size)
        ]
        return self.affine(*matrices, *vectors)[size * size :]


@lru_cache(maxsize=32)
def operation(
    fn: Callable[[Any, Any], Any],
    paths: tuple[str, ...],
    axis: int,
    schedule: ScanSchedule,
) -> ScanOp:
    size = len(paths)

    def combine(*args: mx.array) -> tuple[mx.array, ...]:
        left = tree_unflatten(list(zip(paths, args[:size])))
        right = tree_unflatten(list(zip(paths, args[size:])))
        result = tree_flatten(fn(left, right))
        if tuple(path for path, _ in result) != paths:
            raise ScanContractError("fn must return the structure of elems")
        return tuple(leaf for _, leaf in result)

    return ScanOp(combine, size, axis, schedule)


def associative_scan(
    fn: Callable[[Any, Any], Any],
    elems: Any,
    *,
    reverse: bool = False,
    axis: int = 0,
    schedule: ScanSchedule = DEFAULT_SCAN_SCHEDULE,
) -> Any:
    """Scan ``elems`` along ``axis`` with the associative operation ``fn``."""
    flat = tree_flatten(elems)
    if not flat:
        raise ScanContractError("elems must contain at least one array")
    paths = tuple(path for path, _ in flat)
    leaves = tuple(leaf for _, leaf in flat)
    ndim = leaves[0].ndim
    if not -ndim <= axis < ndim:
        raise ScanContractError(f"axis {axis} is out of range for {ndim} dimensions")
    op = operation(fn, paths, axis % ndim, schedule)
    results = op.reverse(*leaves) if reverse else op(*leaves)
    return tree_unflatten(list(zip(paths, results)))
