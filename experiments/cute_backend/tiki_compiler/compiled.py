"""Own function specialization and compiled forward/reverse derivative callbacks."""

from typing import cast

from mlx import core

from .arrays import arrays, profile, single
from .artifact import CudaIo
from .execution import BackendUnavailableError, launch_kernel, program_executor
from .graph import ArrayFunction, ArrayResult, UnsupportedGraphError
from .lowered import Lowered
from .program import Program
from .schedule import RowSchedule, Schedule, TransposeSchedule
from .specialize import specialize


class Compiled:
    """A specialized array function with registered reverse and forward derivatives.

    Specialize on each input's live layout. Elementwise schedules consume views
    in place; cooperative schedules pack them. Derivatives compile the traced
    VJP and JVP graphs, one kernel per cotangent, through the same lowering.
    """

    def __init__(
        self,
        io: CudaIo,
        function: ArrayFunction,
        schedule: Schedule | RowSchedule | TransposeSchedule,
    ) -> None:
        self.io = io
        self.function = function
        self.schedule = schedule
        self._differentiable = core.custom_function(self._opaque_forward)
        self._differentiable.vjp(self._vjp)
        self._differentiable.jvp(self._jvp)
        self._cotangent_kernels: dict[int, Compiled] = {}
        self._tangent_kernel: Compiled | None = None

    def _opaque_forward(self, *inputs: core.array) -> ArrayResult:
        """Keep raw kernels out of autodiff; registered rules use the original primals."""
        outputs = self.launch(*(core.stop_gradient(value) for value in inputs))
        return outputs

    def lower(self, *inputs: core.array) -> Lowered | Program:
        if not inputs:
            raise UnsupportedGraphError("at least one array input is required")
        if any(
            not isinstance(value, core.array) or value.dtype != core.float32 for value in inputs
        ):
            raise UnsupportedGraphError("all arguments must be float32 MLX arrays")
        lowered = specialize(
            self.function,
            self.schedule,
            tuple(profile(value) for value in inputs),
            core.default_stream(core.default_device()),
        )
        return lowered

    def __call__(self, *inputs: core.array) -> core.array | tuple[core.array, ...]:
        # The native custom_function stub erases the return type of our launch callback.
        result = cast(ArrayResult, self._differentiable(*inputs))
        return result

    def launch(self, *inputs: core.array) -> core.array | tuple[core.array, ...]:
        if not core.cuda.is_available():
            raise BackendUnavailableError("tk.compile execution requires MLX CUDA")
        if core.device_info(core.gpu)["architecture"] != self.schedule.arch:
            raise BackendUnavailableError(f"schedule requires {self.schedule.arch}")
        lowered = self.lower(*inputs)
        if isinstance(lowered, Program):
            if any(stage.stream.device != core.gpu for stage in lowered.stages):
                raise BackendUnavailableError("compiled program stages require GPU streams")
            executor = program_executor(self.io, lowered)
            outputs = executor(*inputs)
        else:
            outputs = launch_kernel(self.io, lowered, inputs, core.default_stream(core.gpu))
        outputs = outputs[0] if len(outputs) == 1 else tuple(outputs)
        return outputs

    def _cotangent_kernel(self, i: int, n: int) -> Compiled:
        """The compiled VJP region for input ``i``; traced once per compiled function."""
        if i not in self._cotangent_kernels:

            def input_cotangent(*args: core.array) -> core.array:
                result = core.vjp(self.function, list(args[:n]), list(args[n:]))[1][i]
                return result

            self._cotangent_kernels[i] = Compiled(
                io=self.io, function=input_cotangent, schedule=self.schedule
            )
        result = self._cotangent_kernels[i]
        return result

    def _vjp(
        self,
        primals: core.array | tuple[core.array, ...],
        cotangents: core.array | tuple[core.array, ...],
        outputs: core.array | tuple[core.array, ...],
    ) -> tuple[core.array, ...]:
        """MLX passes bare arrays for a single-output function, tuples otherwise."""
        primals, cotangents = arrays(primals), arrays(cotangents)
        n = len(primals)
        cotangents = tuple(
            single(self._cotangent_kernel(i, n).launch(*primals, *cotangents)) for i in range(n)
        )
        return cotangents

    def _jvp(
        self,
        primals: core.array | tuple[core.array, ...],
        tangents: core.array | tuple[core.array, ...],
    ) -> core.array | tuple[core.array, ...]:
        """MLX passes (primals, tangents) and expects the output tangents."""
        primals, tangents = arrays(primals), arrays(tangents)
        n = len(primals)
        if self._tangent_kernel is None:

            def output_tangent(*args: core.array) -> tuple[core.array, ...]:
                result = tuple(core.jvp(self.function, list(args[:n]), list(args[n:]))[1])
                return result

            self._tangent_kernel = Compiled(
                io=self.io, function=output_tangent, schedule=self.schedule
            )
        tangents = self._tangent_kernel.launch(*primals, *tangents)
        return tangents
