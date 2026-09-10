"""Execute recursive tiled scans and own their compiled derivative recurrences."""

from functools import lru_cache
from math import prod
from typing import cast

from mlx import core

from .arrays import arrays, profile
from .artifact import CudaIo, binary
from .capture import capture
from .compiled import Compiled
from .execution import BackendUnavailableError
from .graph import Graph, Profile, UnsupportedGraphError
from .scan import lower_apply, lower_tile_scan
from .scan_ad import FlatCombine, affine_combine, basis, flip, matvec, view
from .scan_schedule import ScanLowered, ScanSchedule
from .schedule import Schedule

Leaves = tuple[core.array, ...]


class ScanContractError(ValueError):
    """The leaves are outside the scan contract."""


@lru_cache(maxsize=64)
def tile_kernel(
    combine: Graph, profiles: tuple[Profile, ...], axis: int, schedule: ScanSchedule
) -> ScanLowered:
    result = lower_tile_scan(combine, profiles, axis, schedule)
    return result


@lru_cache(maxsize=64)
def apply_kernel(
    combine: Graph,
    shape: tuple[int, ...],
    axis: int,
    tiles: int,
    schedule: ScanSchedule,
) -> ScanLowered:
    result = lower_apply(combine, shape, axis, tiles, schedule)
    return result


def launch(io: CudaIo, lowered: ScanLowered, inputs: Leaves) -> Leaves:
    if not core.cuda.is_available():
        raise BackendUnavailableError("associative_scan execution requires MLX CUDA")
    if core.device_info(core.gpu)["architecture"] != lowered.schedule.arch:
        raise BackendUnavailableError(f"schedule requires {lowered.schedule.arch}")
    outputs = tuple(
        core.fast.precompiled_cuda_kernel(
            name=lowered.name,
            compiled_source=binary(io, lowered).cubin,
            inputs=list(inputs),
            output_shapes=list(lowered.output_shapes),
            output_dtypes=[core.float32] * len(lowered.output_shapes),
            scalars=[],
            grid=lowered.grid,
            threadgroup=(lowered.schedule.threads, 1, 1),
            shared_memory=lowered.shared_memory_bytes,
            ensure_row_contiguous=False,
            stream=core.gpu,
        )
    )
    return outputs


class ScanOp:
    """An associative scan of ``leaves`` float32 arrays along ``axis``."""

    def __init__(
        self, io: CudaIo, combine: FlatCombine, leaves: int, axis: int, schedule: ScanSchedule
    ) -> None:
        self.io = io
        self.combine = combine
        self.leaves = leaves
        self.axis = axis
        self.schedule = schedule
        self.graph = capture(combine, (Profile(shape=(), strides=()),) * (2 * leaves))
        if len(self.graph.outputs) != leaves:
            raise UnsupportedGraphError("the combine must return one array per leaf")
        self._function = core.custom_function(self._opaque_forward)
        self._function.vjp(self._vjp)
        self._function.jvp(self._jvp)
        self._aggregate: ScanOp | None = None
        self._affine: ScanOp | None = None
        derivative_schedule = Schedule(arch=schedule.arch)
        self._jacobian = Compiled(io=io, function=self.jacobian, schedule=derivative_schedule)
        self._matvec = Compiled(
            io=io, function=matvec(leaves, transpose=False), schedule=derivative_schedule
        )
        self._matvec_transposed = Compiled(
            io=io, function=matvec(leaves, transpose=True), schedule=derivative_schedule
        )

    def _opaque_forward(self, *leaves: core.array) -> Leaves:
        """Keep raw kernels out of autodiff; registered rules use the original primals."""
        outputs = self.forward(*(core.stop_gradient(leaf) for leaf in leaves))
        return outputs

    def __call__(self, *leaves: core.array) -> Leaves:
        # The native custom_function stub erases our forward callback result.
        result = cast(Leaves, self._function(*leaves))
        return result

    def reverse(self, *leaves: core.array) -> Leaves:
        axis = self.axis
        outputs = tuple(
            flip(result, axis) for result in self(*(flip(leaf, axis) for leaf in leaves))
        )
        return outputs

    def check(self, leaves: Leaves) -> tuple[int, ...]:
        if len(leaves) != self.leaves:
            raise ScanContractError(f"expected {self.leaves} leaves, got {len(leaves)}")
        shape = tuple(leaves[0].shape)
        if any(tuple(leaf.shape) != shape or leaf.dtype != core.float32 for leaf in leaves):
            raise ScanContractError("all leaves must be float32 arrays of one shape")
        if not 0 <= self.axis < len(shape):
            raise ScanContractError(f"axis {self.axis} is out of range for shape {shape}")
        if prod(shape) >= 2**31 - 1024:
            raise ScanContractError("element count exceeds signed 32-bit indexing")
        return shape

    def forward(self, *leaves: core.array) -> Leaves:
        shape = self.check(leaves)
        if prod(shape) == 0:
            outputs = tuple(core.zeros(shape, dtype=core.float32, stream=core.gpu) for _ in leaves)
            return outputs
        profiles = tuple(profile(leaf) for leaf in leaves)
        lowered = tile_kernel(self.graph, profiles, self.axis, self.schedule)
        outputs = launch(self.io, lowered, leaves)
        local, aggregates = outputs[: self.leaves], outputs[self.leaves :]
        tiles = lowered.output_shapes[-1][1]
        if tiles == 1:
            return local
        if self._aggregate is None:
            self._aggregate = ScanOp(
                io=self.io, combine=self.combine, leaves=self.leaves, axis=1, schedule=self.schedule
            )
        carry = self._aggregate.forward(*aggregates)
        applied = apply_kernel(self.graph, shape, self.axis, tiles, self.schedule)
        outputs = launch(self.io, applied, (*carry, *local))
        return outputs

    @property
    def affine(self) -> ScanOp:
        """The derivative recurrences as a scan over ``(size x size matrix, size vector)`` pairs."""
        if self._affine is None:
            size = self.leaves
            self._affine = ScanOp(
                io=self.io,
                combine=affine_combine(size),
                leaves=size * size + size,
                axis=self.axis,
                schedule=self.schedule,
            )
        result = self._affine
        return result

    def jacobian(self, *args: core.array) -> tuple[core.array, ...]:
        """``J_y`` then ``J_x`` entries, row-major ``(output row, input column)``, per position."""
        size = self.leaves
        primals = list(args)
        columns = []
        for column in range(2 * size):
            tangents = [basis(primal, float(row == column)) for row, primal in enumerate(primals)]
            columns.append(core.jvp(lambda *a: list(self.combine(*a)), primals, tangents)[1])
        left = [columns[column][row] for row in range(size) for column in range(size)]
        right = [columns[size + column][row] for row in range(size) for column in range(size)]
        jacobians = (*left, *right)
        return jacobians

    def _vjp(
        self,
        primals: core.array | Leaves,
        cotangents: core.array | Leaves,
        outputs: core.array | Leaves,
    ) -> Leaves:
        x, g, y = arrays(primals), arrays(cotangents), arrays(outputs)
        size, axis = self.leaves, self.axis
        length = x[0].shape[axis]
        head = [view(leaf, axis, 0, length - 1) for leaf in y]
        tail = [view(leaf, axis, 1, length) for leaf in x]
        jacobians = arrays(self._jacobian(*head, *tail))
        left, right = jacobians[: size * size], jacobians[size * size :]
        pad = core.zeros_like(view(x[0], axis, 0, 1))
        matrices = [
            core.concatenate([left[column * size + row], pad], axis=axis)
            for row in range(size)
            for column in range(size)
        ]
        gy = self.affine.reverse(*matrices, *g)[size * size :]
        gx_tail = arrays(
            self._matvec_transposed(*right, *(view(leaf, axis, 1, length) for leaf in gy))
        )
        cotangents = tuple(
            core.concatenate([view(gy[column], axis, 0, 1), gx_tail[column]], axis=axis)
            for column in range(size)
        )
        return cotangents

    def _jvp(self, primals: core.array | Leaves, tangents: core.array | Leaves) -> Leaves:
        x, dx = arrays(primals), arrays(tangents)
        size, axis = self.leaves, self.axis
        length = x[0].shape[axis]
        y = self.forward(*x)
        head = [view(leaf, axis, 0, length - 1) for leaf in y]
        tail = [view(leaf, axis, 1, length) for leaf in x]
        jacobians = arrays(self._jacobian(*head, *tail))
        left, right = jacobians[: size * size], jacobians[size * size :]
        pad = core.zeros_like(view(x[0], axis, 0, 1))
        matrices = [core.concatenate([pad, entry], axis=axis) for entry in left]
        driven = arrays(self._matvec(*right, *(view(leaf, axis, 1, length) for leaf in dx)))
        vectors = [
            core.concatenate([view(dx[row], axis, 0, 1), driven[row]], axis=axis)
            for row in range(size)
        ]
        tangents = self.affine(*matrices, *vectors)[size * size :]
        return tangents
