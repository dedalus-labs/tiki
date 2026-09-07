"""Experimental compiler.compile: Tiki export to scheduled CuTe MLIR to CUDA."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory

import tiki as tk

from graph import (
    ArrayFunction,
    ArrayResult,
    Graph,
    Profile,
    Shape,
    UnsupportedGraphError,
    capture,
    dense_strides,
    replay,
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
    """Execution requires the selected Tiki CUDA device."""


def profile(array: tk.array) -> Profile:
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


def _arrays(value: tk.array | tuple[tk.array, ...]) -> tuple[tk.array, ...]:
    """Tiki passes a bare array to derivative callbacks of single-input functions."""
    return (value,) if isinstance(value, tk.array) else tuple(value)


class Compiled:
    """A specialized array function with registered reverse and forward derivatives.

    Elementwise schedules specialize on live input layouts and consume views
    in place. Cooperative schedules pack inputs explicitly. Derivatives use
    the captured forward graph and require a supported derivative schedule.
    """

    def __init__(
        self,
        function: ArrayFunction,
        schedule: Schedule | RowSchedule | TransposeSchedule,
    ):
        self.function = function
        self.schedule = schedule

    def lower(self, *inputs: tk.array) -> Lowered:
        if not inputs:
            raise UnsupportedGraphError("at least one array input is required")
        if any(
            not isinstance(value, tk.array) or value.dtype != tk.float32
            for value in inputs
        ):
            raise UnsupportedGraphError("all arguments must be float32 Tiki arrays")
        return specialize(
            self.function, self.schedule, tuple(profile(value) for value in inputs)
        )

    def __call__(self, *inputs: tk.array) -> ArrayResult:
        return self.launch(*inputs)

    def launch(self, *inputs: tk.array) -> ArrayResult:
        if not tk.cuda.is_available():
            raise BackendUnavailableError("compiler.compile execution requires Tiki CUDA")
        if tk.device_info(tk.gpu)["architecture"] != self.schedule.arch:
            raise BackendUnavailableError(f"schedule requires {self.schedule.arch}")
        lowered = self.lower(*inputs)
        if prod(lowered.graph.shape) == 0:
            outputs = tuple(
                tk.zeros(shape, dtype=tk.float32, stream=tk.gpu)
                for shape in lowered.output_shapes
            )
            return outputs[0] if len(outputs) == 1 else outputs
        return differentiable(lowered)(*inputs)


@lru_cache(maxsize=32)
def differentiable(lowered: Lowered) -> ArrayFunction:
    """The transform tape owns its specialization even after cache eviction."""

    @tk.custom_function
    def forward(*inputs: tk.array) -> ArrayResult:
        shapes = lowered.output_shapes
        outputs = tk.fast.precompiled_cuda_kernel(
            name="tiki_fused",
            compiled_source=binary(lowered).cubin,
            inputs=list(inputs),
            output_shapes=list(shapes),
            output_dtypes=[tk.float32] * len(shapes),
            scalars=[],
            grid=lowered.grid,
            threadgroup=(lowered.schedule.threads, 1, 1),
            shared_memory=lowered.shared_memory_bytes,
            ensure_row_contiguous=packs_views(lowered.schedule),
            stream=tk.gpu,
        )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    @forward.vjp
    def vjp(
        primals: tk.array | tuple[tk.array, ...],
        cotangents: ArrayResult,
        outputs: ArrayResult,
    ) -> tuple[tk.array, ...]:
        """Tiki passes one cotangent per output, a bare array for one output."""
        primals, cotangents = _arrays(primals), _arrays(cotangents)
        return tuple(
            derivative(lowered.graph, lowered.schedule, input_index)(
                *primals, *cotangents
            )
            for input_index in range(len(primals))
        )

    @forward.jvp
    def jvp(
        primals: tk.array | tuple[tk.array, ...],
        tangents: tk.array | tuple[tk.array, ...],
    ) -> ArrayResult:
        """Tiki expects all output tangents in the forward result convention."""
        primals, tangents = _arrays(primals), _arrays(tangents)
        return derivative(lowered.graph, lowered.schedule, None)(*primals, *tangents)

    return forward


@lru_cache(maxsize=32)
def derivative(
    graph: Graph,
    schedule: Schedule | RowSchedule | TransposeSchedule,
    index: int | None,
) -> Compiled:
    """Cache derivative programs by the exact forward graph, including its arity."""
    input_count = len(graph.inputs)

    def frozen(*inputs: tk.array) -> tuple[tk.array, ...]:
        return replay(graph, inputs)

    def differentiate(*args: tk.array) -> ArrayResult:
        primals, derivatives = args[:input_count], args[input_count:]
        if index is None:
            return tuple(tk.jvp(frozen, primals, derivatives)[1])
        return tk.vjp(frozen, primals, derivatives)[1][index]

    return Compiled(differentiate, schedule)


DEFAULT_SCHEDULE = Schedule()


def compile(
    *,
    backend: str = "cute",
    schedule: Schedule | RowSchedule | TransposeSchedule = DEFAULT_SCHEDULE,
) -> Callable[[ArrayFunction], Compiled]:
    """Specialize a pure array function, freezing captures per layout profile."""
    if backend != "cute":
        raise UnsupportedScheduleError(f"unsupported backend: {backend}")
    return lambda function: Compiled(function, schedule)
