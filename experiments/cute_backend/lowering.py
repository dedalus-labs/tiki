"""Lower an explicit elementwise schedule directly into CuTe MLIR."""

import struct
from dataclasses import dataclass
from math import prod

from graph import Graph, Node, Value


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
class Lowered:
    graph: Graph
    schedule: Schedule
    mlir: str

    @property
    def grid(self) -> tuple[int, int, int]:
        tile = self.schedule.threads * self.schedule.elements_per_thread
        blocks = (prod(self.graph.shape) + tile - 1) // tile
        return (blocks * self.schedule.threads, 1, 1)


def memref(value: Value) -> str:
    return f'!cute.memref<f32, gmem, align<4>, "({prod(value.shape)}):(1)">'


def expression(node: Node, names: dict[str, str]) -> str:
    args = [names[name] for name in node.inputs]
    if node.operation == "Broadcast":
        return args[0]
    if node.operation == "Square":
        return f"arith.mulf {args[0]}, {args[0]} : f32"
    if node.operation == "Negative":
        return f"arith.negf {args[0]} : f32"
    opcode = {"Add": "addf", "Subtract": "subf", "Multiply": "mulf"}[node.operation]
    return f"arith.{opcode} {args[0]}, {args[1]} : f32"


def element(graph: Graph, index: int) -> list[str]:
    lines = [f'%coord = cute.make_coord(%index{index}) : (i32) -> !cute.coord<"?">']
    lines.append("%zero = arith.constant 0 : i32")
    lines.append('%scalar = cute.make_coord(%zero) : (i32) -> !cute.coord<"?">')
    names = {}
    for i, value in enumerate(graph.inputs):
        coord = "%scalar" if value.shape == () else "%coord"
        names[value.name] = f"%input{i}"
        lines.append(
            f'%input{i} = cute.memref.load(%arg{i}, {coord}) : ({memref(value)}, !cute.coord<"?">) -> f32'
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
    output = Value(graph.output, graph.shape)
    lines.append(
        f'cute.memref.store(%arg{len(graph.inputs)}, %coord, {names[graph.output]}) : ({memref(output)}, !cute.coord<"?">, f32) -> ()'
    )
    return lines


def lower(graph: Graph, schedule: Schedule) -> Lowered:
    if prod(graph.shape) == 0:
        return Lowered(graph, schedule, "module {}\n")
    values = (*graph.inputs, Value(graph.output, graph.shape))
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
