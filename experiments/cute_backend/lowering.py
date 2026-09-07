"""Lower an explicit elementwise schedule directly into CuTe MLIR."""

import struct
from dataclasses import dataclass
from math import prod

from mlx.tiki import Swizzle

from graph import Graph, Node, Shape, UnsupportedGraphError, Value
from operations import Operation


class UnsupportedScheduleError(ValueError):
    """The requested schedule is outside the supported CUDA target."""


@dataclass(frozen=True)
class Schedule:
    arch: str = "sm_90"
    threads: int = 128
    elements_per_thread: int = 1

    def __post_init__(self) -> None:
        if self.arch != "sm_90":
            raise UnsupportedScheduleError("only sm_90 is validated")
        if type(self.threads) is not int or type(self.elements_per_thread) is not int:
            raise UnsupportedScheduleError("schedule dimensions must be integers")
        if self.threads not in (32, 64, 128, 256):
            raise UnsupportedScheduleError("threads must be 32, 64, 128, or 256")
        if self.elements_per_thread not in (1, 2, 4):
            raise UnsupportedScheduleError("elements_per_thread must be 1, 2, or 4")


@dataclass(frozen=True)
class RowSchedule:
    arch: str = "sm_90"
    threads_per_row: int = 128
    rows_per_block: int = 1

    def __post_init__(self) -> None:
        if type(self.rows_per_block) is not int or self.rows_per_block not in (
            1,
            2,
            4,
            8,
            16,
            32,
        ):
            raise UnsupportedScheduleError(
                "rows_per_block must be a power of two from 1 to 32"
            )
        if type(self.threads_per_row) is not int or self.threads_per_row not in (
            8,
            16,
            32,
            64,
            128,
            256,
        ):
            raise UnsupportedScheduleError(
                "threads_per_row must be 8, 16, 32, 64, 128, or 256"
            )
        Schedule(arch=self.arch, threads=self.threads)

    @property
    def threads(self) -> int:
        return self.threads_per_row * self.rows_per_block


@dataclass(frozen=True)
class TransposeSchedule:
    arch: str = "sm_90"
    threads: int = 128
    swizzle: Swizzle = Swizzle(5, 0, 5)

    def __post_init__(self) -> None:
        Schedule(arch=self.arch, threads=self.threads)
        if not isinstance(self.swizzle, Swizzle):
            raise UnsupportedScheduleError(
                "transpose requires a validated Tiki Swizzle"
            )
        if self.swizzle.bits + self.swizzle.base > 5 or self.swizzle.shift != 5:
            raise UnsupportedScheduleError(
                "32x32 transpose requires bits + base <= 5 and shift=5"
            )


@dataclass(frozen=True)
class Lowered:
    graph: Graph
    schedule: Schedule | RowSchedule | TransposeSchedule
    mlir: str

    @property
    def grid(self) -> tuple[int, int, int]:
        if isinstance(self.schedule, RowSchedule):
            rows = self.graph.shape[0]
            blocks = (
                rows + self.schedule.rows_per_block - 1
            ) // self.schedule.rows_per_block
            return (blocks * self.schedule.threads, 1, 1)
        if isinstance(self.schedule, TransposeSchedule):
            rows, cols = self.graph.shape
            blocks = ((rows + 31) // 32) * ((cols + 31) // 32)
            return (blocks * self.schedule.threads, 1, 1)
        tile = self.schedule.threads * self.schedule.elements_per_thread
        blocks = (prod(self.graph.shape) + tile - 1) // tile
        return (blocks * self.schedule.threads, 1, 1)

    @property
    def output_shapes(self) -> tuple[Shape, ...]:
        return tuple(value.shape for value in self.graph.outputs)

    @property
    def shared_memory_bytes(self) -> int:
        if isinstance(self.schedule, RowSchedule):
            if not any(node.operation == "ReduceSum" for node in self.graph.nodes):
                return 0
            warps = self.schedule.threads_per_row // 32
            return 4 * self.schedule.rows_per_block * warps if warps > 1 else 0
        if isinstance(self.schedule, TransposeSchedule):
            return 4096
        return 0


def memref(value: Value) -> str:
    """Dense values keep the flat form; a strided view addresses through its own layout."""
    if value.is_dense:
        return f'!cute.memref<f32, gmem, align<4>, "({prod(value.shape)}):(1)">'
    shape = ",".join(str(extent) for extent in value.shape)
    strides = ",".join(str(stride) for stride in value.strides)
    return f'!cute.memref<f32, gmem, align<4>, "({shape}):({strides})">'


def logical_coordinate(shape: Shape, index: int) -> list[str]:
    """Split the flat right-major output index into one index per axis."""
    lines = [f"%rem0 = arith.addi %index{index}, %zero : i32"]
    axes = len(shape)
    for k in range(axes - 1, -1, -1):
        lines.append(f"%extent{k} = arith.constant {shape[k]} : i32")
        if k > 0:
            lines.append(f"%i{k} = arith.remsi %rem{axes - 1 - k}, %extent{k} : i32")
            lines.append(
                f"%rem{axes - k} = arith.divsi %rem{axes - 1 - k}, %extent{k} : i32"
            )
        else:
            lines.append(f"%i0 = arith.addi %rem{axes - 1}, %zero : i32")
    args = ", ".join(f"%i{k}" for k in range(axes))
    types = ", ".join("i32" for _ in shape)
    marks = ",".join("?" for _ in shape)
    lines.append(
        f'%logical = cute.make_coord({args}) : ({types}) -> !cute.coord<"({marks})">'
    )
    return lines


def expression(node: Node, names: dict[str, str]) -> str:
    return Operation.require(node.operation).expression(
        tuple(names[name] for name in node.inputs)
    )


def element(graph: Graph, index: int) -> list[str]:
    lines = [f'%coord = cute.make_coord(%index{index}) : (i32) -> !cute.coord<"?">']
    lines.append("%zero = arith.constant 0 : i32")
    lines.append('%scalar = cute.make_coord(%zero) : (i32) -> !cute.coord<"?">')
    strided = any(not value.is_dense for value in graph.inputs)
    if strided:
        lines.extend(logical_coordinate(graph.shape, index))
    marks = ",".join("?" for _ in graph.shape)
    names = {}
    for i, value in enumerate(graph.inputs):
        if value.shape == ():
            coord, coord_type = "%scalar", '!cute.coord<"?">'
        elif value.is_dense:
            coord, coord_type = "%coord", '!cute.coord<"?">'
        else:
            coord, coord_type = "%logical", f'!cute.coord<"({marks})">'
        names[value.name] = f"%input{i}"
        lines.append(
            f"%input{i} = cute.memref.load(%arg{i}, {coord}) : ({memref(value)}, {coord_type}) -> f32"
        )
    for i, (name, value) in enumerate(graph.constants):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        names[name] = f"%constant{i}"
        lines.append(f"%constant{i} = arith.constant 0x{bits:08X} : f32")
    for i, node in enumerate(graph.nodes):
        result = expression(node, names)
        if node.operation == "Broadcast":
            names[node.output.name] = result
            continue
        names[node.output.name] = f"%value{i}"
        lines.append(f"%value{i} = {result}")
    for j, output in enumerate(graph.outputs):
        lines.append(
            f'cute.memref.store(%arg{len(graph.inputs) + j}, %coord, {names[output.name]}) : ({memref(output)}, !cute.coord<"?">, f32) -> ()'
        )
    return lines


def lower(graph: Graph, schedule: Schedule) -> Lowered:
    if any(value.shape not in ((), graph.shape) for value in graph.inputs):
        raise UnsupportedGraphError(
            "elementwise inputs must have the output shape or be scalars"
        )
    if any(value.shape != graph.shape for value in graph.outputs):
        raise UnsupportedGraphError("elementwise outputs must share one shape")
    for node in graph.nodes:
        if node.operation in ("ReduceSum", "Transpose") or node.output.shape not in (
            (),
            graph.shape,
        ):
            raise UnsupportedGraphError(
                f"unsupported elementwise node: {node.operation}"
            )
    if prod(graph.shape) == 0:
        return Lowered(graph, schedule, "module {}\n")
    values = (*graph.inputs, *graph.outputs)
    parameters = ", ".join(
        f"%arg{i}: {memref(value)}" for i, value in enumerate(values)
    )
    lines = [
        "module attributes {gpu.container_module} {",
        "  gpu.module @kernels {",
        f"    cuda.kernel @tiki_fused({parameters}) attributes {{cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: {schedule.threads}, 1, 1>}} {{",
        "      %thread = nvvm.read.ptx.sreg.tid.x : i32",
        "      %block = nvvm.read.ptx.sreg.ctaid.x : i32",
        f"      %tile = arith.constant {schedule.threads * schedule.elements_per_thread} : i32",
        f"      %size = arith.constant {prod(graph.shape)} : i32",
        "      %base = arith.muli %block, %tile : i32",
        "      %first = arith.addi %base, %thread : i32",
    ]
    for i in range(schedule.elements_per_thread):
        lines.extend(
            [
                f"      %offset{i} = arith.constant {i * schedule.threads} : i32",
                f"      %index{i} = arith.addi %first, %offset{i} : i32",
                f"      %valid{i} = arith.cmpi slt, %index{i}, %size : i32",
                f"      scf.if %valid{i} {{",
                *("        " + line for line in element(graph, i)),
                "      }",
            ]
        )
    lines.extend(["      return", "    }", "  }", "}"])
    return Lowered(graph, schedule, "\n".join(lines) + "\n")
