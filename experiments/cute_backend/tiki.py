"""Experimental tk.compile: MLX export to scheduled CuTe MLIR to CUDA."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from graph import ArrayFunction, Shape, UnsupportedGraphError, capture
from lowering import Lowered, Schedule, UnsupportedScheduleError, lower


class BackendUnavailableError(RuntimeError):
    """Execution requires the selected MLX CUDA device."""


@lru_cache(maxsize=32)
def specialize(
    function: ArrayFunction, schedule: Schedule, shapes: tuple[Shape, ...]
) -> Lowered:
    graph = capture(function, shapes)
    return lower(graph, schedule)


@lru_cache(maxsize=32)
def binary(lowered: Lowered) -> bytes:
    from cutlass import compiler

    from compile_mlir import cute_pipeline

    artifact = compiler.PreCompiledMlirArtifact.from_textual_form(lowered.mlir.encode())
    backend = compiler.CuteCompiler()
    backend.set_device_target(lowered.schedule.arch)
    backend.set_abi(compiler.Abi.Tbd)
    with TemporaryDirectory(prefix="tiki-compile-") as directory:
        prefix = Path(directory) / "kernel"
        backend.set_pipeline(
            compiler.ArtifactType.PreCompiledMlir, cute_pipeline(prefix)
        )
        backend.compile_to(artifact, compiler.ArtifactType.Object)
        return (Path(directory) / f"kernel.{lowered.schedule.arch}.cubin").read_bytes()


@dataclass(frozen=True)
class Compiled:
    function: ArrayFunction
    schedule: Schedule

    def lower(self, *inputs: mx.array) -> Lowered:
        if not inputs:
            raise UnsupportedGraphError("at least one array input is required")
        if any(
            not isinstance(value, mx.array) or value.dtype != mx.float32
            for value in inputs
        ):
            raise UnsupportedGraphError("all arguments must be float32 MLX arrays")
        return specialize(
            self.function, self.schedule, tuple(value.shape for value in inputs)
        )

    def __call__(self, *inputs: mx.array) -> mx.array:
        if not mx.cuda.is_available():
            raise BackendUnavailableError("tk.compile execution requires MLX CUDA")
        if mx.device_info(mx.gpu)["architecture"] != self.schedule.arch:
            raise BackendUnavailableError(f"schedule requires {self.schedule.arch}")
        lowered = self.lower(*inputs)
        if prod(lowered.graph.shape) == 0:
            return mx.zeros(lowered.graph.shape, dtype=mx.float32, stream=mx.gpu)
        return mx.fast.precompiled_cuda_kernel(
            name="tiki_fused",
            compiled_source=binary(lowered),
            inputs=list(inputs),
            output_shapes=[lowered.graph.shape],
            output_dtypes=[mx.float32],
            scalars=[],
            grid=lowered.grid,
            threadgroup=(self.schedule.threads, 1, 1),
            ensure_row_contiguous=True,
            stream=mx.gpu,
        )[0]


DEFAULT_SCHEDULE = Schedule()


def compile(
    *, backend: str = "cute", schedule: Schedule = DEFAULT_SCHEDULE
) -> Callable[[ArrayFunction], Compiled]:
    """Specialize a pure array function; captured Python values are frozen per shape."""
    if backend != "cute":
        raise UnsupportedScheduleError(f"unsupported backend: {backend}")
    return lambda function: Compiled(function, schedule)
