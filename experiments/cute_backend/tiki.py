"""Experimental tk.compile: MLX export to scheduled CuTe MLIR to CUDA."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from graph import (
    ArrayFunction,
    Profile,
    Shape,
    UnsupportedGraphError,
    capture,
    dense_strides,
)
from lowering import (  # noqa: F401 - Public schedule types.
    Lowered,
    RowSchedule,
    Schedule,
    Swizzle,
    TransposeSchedule,
    UnsupportedScheduleError,
    lower,
)


class BackendUnavailableError(RuntimeError):
    """Execution requires the selected MLX CUDA device."""


def profile(array: mx.array) -> Profile:
    """Shape and strides of an evaluated array; this is the specialization key."""
    return tuple(array.shape), tuple(array.strides)


def packs_views(schedule: Schedule | RowSchedule | TransposeSchedule) -> bool:
    """Cooperative schedules address dense storage only, so their views are
    packed to row-major before launch and they specialize on the dense
    profile. The elementwise schedule addresses each view in place."""
    return not isinstance(schedule, Schedule)


@lru_cache(maxsize=32)
def specialize(
    function: ArrayFunction,
    schedule: Schedule | RowSchedule | TransposeSchedule,
    profiles: tuple[Profile, ...],
) -> Lowered:
    if packs_views(schedule):
        profiles = tuple((shape, dense_strides(shape)) for shape, _ in profiles)
    graph = capture(function, profiles)
    if isinstance(schedule, RowSchedule):
        from row_reduction import lower_row

        return lower_row(graph, schedule)
    if isinstance(schedule, TransposeSchedule):
        from transpose import lower_transpose

        return lower_transpose(graph, schedule)
    return lower(graph, schedule)


@dataclass(frozen=True)
class CudaBinary:
    cubin: bytes
    ptx: str


@lru_cache(maxsize=32)
def binary(lowered: Lowered) -> CudaBinary:
    from cutlass import compiler

    from compile_mlir import cute_pipeline

    artifact = compiler.PreCompiledMlirArtifact.from_textual_form(lowered.mlir.encode())
    backend = compiler.CuteCompiler()
    backend.set_device_target(lowered.schedule.arch)
    backend.set_abi(compiler.Abi.Tbd)
    with TemporaryDirectory(prefix="tiki-compile-") as directory:
        prefix = Path(directory) / "kernel"
        backend.set_pipeline(
            compiler.ArtifactType.PreCompiledMlir, cute_pipeline(prefix, keep_ptx=True)
        )
        backend.compile_to(artifact, compiler.ArtifactType.Object)
        return CudaBinary(
            (Path(directory) / f"kernel.{lowered.schedule.arch}.cubin").read_bytes(),
            (Path(directory) / f"kernel.{lowered.schedule.arch}.ptx").read_text(),
        )


def _arrays(value: mx.array | tuple[mx.array, ...]) -> tuple[mx.array, ...]:
    """MLX passes a bare array to derivative callbacks of single-input functions."""
    return (value,) if isinstance(value, mx.array) else tuple(value)


class Compiled:
    """A specialized array function with registered reverse and forward derivatives.

    Each call specializes on the live layout profile of every input, so a
    transposed or sliced view gets its own kernel and is consumed in place:
    nothing is packed. The derivatives are compiled regions of the traced
    VJP and JVP graphs, one kernel per cotangent, lowered the same way.
    """

    def __init__(
        self,
        function: ArrayFunction,
        schedule: Schedule | RowSchedule | TransposeSchedule,
    ):
        self.function = function
        self.schedule = schedule
        self._differentiable = mx.custom_function(self.launch)
        self._differentiable.vjp(self._vjp)
        self._differentiable.jvp(self._jvp)
        self._cotangent_kernels: dict[int, Compiled] = {}
        self._tangent_kernel: Compiled | None = None

    def lower(self, *inputs: mx.array) -> Lowered:
        if not inputs:
            raise UnsupportedGraphError("at least one array input is required")
        if any(
            not isinstance(value, mx.array) or value.dtype != mx.float32
            for value in inputs
        ):
            raise UnsupportedGraphError("all arguments must be float32 MLX arrays")
        return specialize(
            self.function, self.schedule, tuple(profile(value) for value in inputs)
        )

    def __call__(self, *inputs: mx.array) -> mx.array:
        return self._differentiable(*inputs)

    def launch(self, *inputs: mx.array) -> mx.array:
        if not mx.cuda.is_available():
            raise BackendUnavailableError("tk.compile execution requires MLX CUDA")
        if mx.device_info(mx.gpu)["architecture"] != self.schedule.arch:
            raise BackendUnavailableError(f"schedule requires {self.schedule.arch}")
        lowered = self.lower(*inputs)
        shapes = lowered.output_shapes
        if prod(lowered.graph.shape) == 0:
            outputs = [
                mx.zeros(shape, dtype=mx.float32, stream=mx.gpu) for shape in shapes
            ]
        else:
            outputs = mx.fast.precompiled_cuda_kernel(
                name="tiki_fused",
                compiled_source=binary(lowered).cubin,
                inputs=list(inputs),
                output_shapes=list(shapes),
                output_dtypes=[mx.float32] * len(shapes),
                scalars=[],
                grid=lowered.grid,
                threadgroup=(self.schedule.threads, 1, 1),
                shared_memory=lowered.shared_memory_bytes,
                ensure_row_contiguous=packs_views(self.schedule),
                stream=mx.gpu,
            )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _cotangent_kernel(self, i: int, n: int) -> "Compiled":
        """The compiled VJP region for input ``i``; traced once per compiled function."""
        if i not in self._cotangent_kernels:

            def input_cotangent(*args: mx.array) -> mx.array:
                return mx.vjp(self.function, list(args[:n]), list(args[n:]))[1][i]

            self._cotangent_kernels[i] = Compiled(input_cotangent, self.schedule)
        return self._cotangent_kernels[i]

    def _vjp(
        self,
        primals: mx.array | tuple[mx.array, ...],
        cotangents: mx.array | tuple[mx.array, ...],
        outputs: mx.array | tuple[mx.array, ...],
    ) -> tuple[mx.array, ...]:
        """MLX passes bare arrays for a single-output function, tuples otherwise."""
        primals, cotangents = _arrays(primals), _arrays(cotangents)
        n = len(primals)
        return tuple(
            self._cotangent_kernel(i, n).launch(*primals, *cotangents) for i in range(n)
        )

    def _jvp(
        self,
        primals: mx.array | tuple[mx.array, ...],
        tangents: mx.array | tuple[mx.array, ...],
    ) -> mx.array | tuple[mx.array, ...]:
        """MLX passes (primals, tangents) and expects the output tangents."""
        primals, tangents = _arrays(primals), _arrays(tangents)
        n = len(primals)
        if self._tangent_kernel is None:

            def output_tangent(*args: mx.array) -> tuple[mx.array, ...]:
                return tuple(mx.jvp(self.function, list(args[:n]), list(args[n:]))[1])

            self._tangent_kernel = Compiled(output_tangent, self.schedule)
        return self._tangent_kernel.launch(*primals, *tangents)


DEFAULT_SCHEDULE = Schedule()


def compile(
    *,
    backend: str = "cute",
    schedule: Schedule | RowSchedule | TransposeSchedule = DEFAULT_SCHEDULE,
) -> Callable[[ArrayFunction], Compiled]:
    """Specialize a pure array function; captured Python values are frozen per shape."""
    if backend != "cute":
        raise UnsupportedScheduleError(f"unsupported backend: {backend}")
    return lambda function: Compiled(function, schedule)
