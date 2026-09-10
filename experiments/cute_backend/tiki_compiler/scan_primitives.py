"""Emit scan layouts, scalar combines, and lane/warp prefix propagation."""

import struct

from .graph import Graph, Shape, Strides, UnsupportedGraphError
from .scalar import expression
from .scan_schedule import ScanSchedule

COORD = '!cute.coord<"(?,?)">'
INDENT = "      "


def axis_layout(shape: Shape, strides: Strides, axis: int) -> str:
    """``((1, batch...), n):((0, batch strides...), axis stride)``.

    The leading extent-1 mode keeps the batch mode a tuple for rank-1 arrays;
    every memref of one kernel uses the same batch shape, so the flattened row
    coordinate names the same batch position in each of them.
    """
    batch = [
        (extent, stride)
        for k, (extent, stride) in enumerate(zip(shape, strides, strict=True))
        if k != axis
    ]
    extents = ",".join(str(extent) for extent in (1, *(extent for extent, _ in batch)))
    offsets = ",".join(str(stride) for stride in (0, *(stride for _, stride in batch)))
    layout = f"(({extents}),{shape[axis]}):(({offsets}),{strides[axis]})"
    return layout


def gmem(layout: str) -> str:
    memory_type = f'!cute.memref<f32, gmem, align<4>, "{layout}">'
    return memory_type


def load(memory: str, kind: str, coord: str, result: str) -> str:
    instruction = f"{result} = cute.memref.load({memory}, {coord}) : ({kind}, {COORD}) -> f32"
    return instruction


def store(memory: str, kind: str, coord: str, value: str) -> str:
    instruction = f"cute.memref.store({memory}, {coord}, {value}) : ({kind}, {COORD}, f32) -> ()"
    return instruction


def validate_combine(graph: Graph) -> int:
    """The combine is traced on scalars: 2k inputs, k outputs, elementwise nodes."""
    leaves = len(graph.outputs)
    if len(graph.inputs) != 2 * leaves:
        raise UnsupportedGraphError("a combine takes two operands with the output structure")
    values = (*graph.inputs, *graph.outputs, *(node.output for node in graph.nodes))
    if any(value.shape != () for value in values):
        raise UnsupportedGraphError("the combine must be traced on scalars")
    if any(node.operation in ("ReduceSum", "Transpose") for node in graph.nodes):
        raise UnsupportedGraphError("the combine must be elementwise")
    return leaves


class Combine:
    """Emit the combine graph on SSA scalars with names unique per application."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.leaves = validate_combine(graph)
        self.applications = 0

    def constants(self) -> list[str]:
        lines = []
        for index, (_, value) in enumerate(self.graph.constants):
            bits = int.from_bytes(struct.pack("<f", value), "little")
            lines.append(f"%k{index} = arith.constant 0x{bits:08X} : f32")
        return lines

    def apply(self, left: list[str], right: list[str]) -> tuple[list[str], list[str]]:
        names = {
            value.name: ssa for value, ssa in zip(self.graph.inputs, (*left, *right), strict=True)
        }
        names.update((name, f"%k{leaf}") for leaf, (name, _) in enumerate(self.graph.constants))
        self.applications += 1
        lines = []
        for index, node in enumerate(self.graph.nodes):
            result = expression(node, names)
            if node.operation == "Broadcast":
                names[node.output.name] = result
                continue
            names[node.output.name] = f"%c{self.applications}_{index}"
            lines.append(f"{names[node.output.name]} = {result}")
        applied = lines, [names[output.name] for output in self.graph.outputs]
        return applied


def kernel(name: str, params: list[str], threads: int, body: list[str]) -> str:
    lines = [
        "module attributes {gpu.container_module} {",
        "  gpu.module @kernels {",
        f"    cuda.kernel @{name}({', '.join(params)}) attributes "
        f"{{cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: {threads}, 1, 1>}} {{",
        *(INDENT + line for line in body),
        INDENT + "return",
        "    }",
        "  }",
        "}",
    ]
    mlir = "\n".join(lines) + "\n"
    return mlir


def prologue(shape: Shape, axis: int, tiles: int, schedule: ScanSchedule) -> list[str]:
    lines = [
        "%thread = nvvm.read.ptx.sreg.tid.x : i32",
        "%block = nvvm.read.ptx.sreg.ctaid.x : i32",
        f"%tiles = arith.constant {tiles} : i32",
        "%tile = arith.remui %block, %tiles : i32",
        "%row = arith.divui %block, %tiles : i32",
        f"%tile_size = arith.constant {schedule.tile} : i32",
        f"%chunk = arith.constant {schedule.elements_per_thread} : i32",
        f"%length = arith.constant {shape[axis]} : i32",
        "%zero = arith.constant 0 : i32",
        "%one = arith.constant 1 : i32",
        "%tile_base = arith.muli %tile, %tile_size : i32",
        "%thread_offset = arith.muli %thread, %chunk : i32",
        "%chunk_base = arith.addi %tile_base, %thread_offset : i32",
    ]
    return lines


def warp_scan(combine: Combine, values: list[str]) -> tuple[list[str], list[str]]:
    """Inclusive scan of one value per lane; lanes below the shift keep their own value."""
    lines = []
    for shift in (1, 2, 4, 8, 16):
        lines.append(f"%shift{shift} = arith.constant {shift} : i32")
        shifted = []
        for leaf, value in enumerate(values):
            shifted.append(f"%up{leaf}_{shift}")
            lines.append(
                f"{shifted[leaf]} = nvvm.shfl.sync up %mask, {value}, "
                f"%shift{shift}, %clamp : f32 -> f32"
            )
        lines.append(f"%has{shift} = arith.cmpi uge, %lane, %shift{shift} : i32")
        combined, results = combine.apply(shifted, values)
        lines.extend(combined)
        for leaf, value in enumerate(values):
            lines.append(
                f"%warp{leaf}_{shift} = arith.select %has{shift}, {results[leaf]}, {value} : f32"
            )
        values = [f"%warp{leaf}_{shift}" for leaf in range(len(values))]
    scanned = lines, values
    return scanned


def block_prefix(
    combine: Combine, schedule: ScanSchedule, totals: list[str], exclusive: list[str]
) -> tuple[list[str], list[str], str]:
    """Exclusive prefix of every thread from the warp totals staged in shared memory."""
    leaves = combine.leaves
    kind = f'!cute.memref<f32, smem, align<4>, "({schedule.warps},{leaves}):({leaves},1)">'
    lines = [
        f"%shared = cute.memref.alloca() : {kind}",
        "%lane_last = arith.constant 31 : i32",
    ]
    lines.append("%is_last = arith.cmpi eq, %lane, %lane_last : i32")
    for leaf in range(leaves):
        lines.append(f"%slot{leaf} = arith.constant {leaf} : i32")
    lines.append("scf.if %is_last {")
    for leaf in range(leaves):
        lines.append(
            f"  %stage{leaf} = cute.make_coord(%warp, %slot{leaf}) : (i32, i32) -> {COORD}"
        )
        lines.append("  " + store("%shared", kind, f"%stage{leaf}", totals[leaf]))
    lines.extend(["}", "nvvm.barrier"])
    for leaf in range(leaves):
        lines.append(
            f"%first_coord{leaf} = cute.make_coord(%zero, %slot{leaf}) : (i32, i32) -> {COORD}"
        )
        lines.append(load("%shared", kind, f"%first_coord{leaf}", f"%first{leaf}"))
    iterated = ", ".join(f"%fold{leaf} = %first{leaf}" for leaf in range(leaves))
    types = ", ".join("f32" for _ in range(leaves))
    lines.append(
        f"%carry:{leaves} = scf.for %j = %one to %warp step %one "
        f"iter_args({iterated}) -> ({types}) : i32 {{"
    )
    for leaf in range(leaves):
        lines.append(
            f"  %next_coord{leaf} = cute.make_coord(%j, %slot{leaf}) : (i32, i32) -> {COORD}"
        )
        lines.append("  " + load("%shared", kind, f"%next_coord{leaf}", f"%next{leaf}"))
    folded, results = combine.apply(
        [f"%fold{leaf}" for leaf in range(leaves)],
        [f"%next{leaf}" for leaf in range(leaves)],
    )
    lines.extend("  " + line for line in folded)
    lines.append(f"  scf.yield {', '.join(results)} : {types}")
    lines.append("}")
    lines.append("%has_warp = arith.cmpi uge, %warp, %one : i32")
    carry = [f"%carry#{leaf}" for leaf in range(leaves)]
    both, results = combine.apply(carry, exclusive)
    lines.extend(both)
    prefix = []
    for leaf in range(leaves):
        lines.append(
            f"%in_warp{leaf} = arith.select %has_lane, {results[leaf]}, {carry[leaf]} : f32"
        )
        lines.append(
            f"%prefix{leaf} = arith.select %has_warp, %in_warp{leaf}, {exclusive[leaf]} : f32"
        )
        prefix.append(f"%prefix{leaf}")
    lines.append("%has_prefix = arith.ori %has_warp, %has_lane : i1")
    prefix_result = lines, prefix, "%has_prefix"
    return prefix_result
