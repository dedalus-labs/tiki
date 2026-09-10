"""Emit one fused elementwise region, preserving strided input layouts."""

import struct
from math import prod

from .graph import Graph, UnsupportedGraphError
from .lowered import Lowered
from .scalar import expression, logical_coordinate, memref
from .schedule import Schedule


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
            f"%input{i} = cute.memref.load(%arg{i}, {coord}) : "
            f"({memref(value)}, {coord_type}) -> f32"
        )
    for i, (name, value) in enumerate(graph.constants):
        bits = int.from_bytes(struct.pack("<f", value), "little")
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
            f"cute.memref.store(%arg{len(graph.inputs) + j}, %coord, "
            f'{names[output.name]}) : ({memref(output)}, !cute.coord<"?">, f32) -> ()'
        )
    return lines


def lower(graph: Graph, schedule: Schedule) -> Lowered:
    if any(value.shape not in ((), graph.shape) for value in graph.inputs):
        raise UnsupportedGraphError("elementwise inputs must have the output shape or be scalars")
    if any(value.shape != graph.shape for value in graph.outputs):
        raise UnsupportedGraphError("elementwise outputs must share one shape")
    for node in graph.nodes:
        if node.operation in ("ReduceSum", "Transpose") or node.output.shape not in (
            (),
            graph.shape,
        ):
            raise UnsupportedGraphError(f"unsupported elementwise node: {node.operation}")
    if prod(graph.shape) == 0:
        lowered = Lowered(graph=graph, schedule=schedule, mlir="module {}\n")
        return lowered
    values = (*graph.inputs, *graph.outputs)
    parameters = ", ".join(f"%arg{i}: {memref(value)}" for i, value in enumerate(values))
    lines = [
        "module attributes {gpu.container_module} {",
        "  gpu.module @kernels {",
        f"    cuda.kernel @tiki_fused({parameters}) attributes "
        f"{{cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: {schedule.threads}, 1, 1>}} {{",
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
    lowered = Lowered(graph=graph, schedule=schedule, mlir="\n".join(lines) + "\n")
    return lowered
