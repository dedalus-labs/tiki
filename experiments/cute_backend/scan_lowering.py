"""Lower an associative scan to tiled CuTe kernels.

The scan axis is split into tiles of ``threads * elements_per_thread``
positions. Within a tile the scan has three levels, the structure of Triton's
ScanOp lowering: each thread folds its contiguous chunk in registers, the
chunk totals are scanned across the warp with shuffles, and the warp totals
are scanned through shared memory. The tile kernel writes the tile-local scan
and one aggregate per tile; the aggregates are scanned recursively at the graph
level and folded back by the apply kernel. No level needs an identity element:
a lane, warp, or tile with no left neighbor keeps its own value, loads past
the row are clamped to its last position, and stores are predicated, so every
length takes the same path. The combine is the captured graph applied to SSA
scalars, so any elementwise combine over any number of leaves lowers here.

Every array is addressed through a two-mode CuTe layout ``((1, batch...),
n)`` whose row coordinate flattens the batch axes and whose second coordinate
is the scan axis; strided views are consumed in place, and a reversed view
(negative axis stride) makes a reverse scan a forward scan.
"""

from dataclasses import dataclass
from math import prod

from graph import Graph, Profile, Shape, Strides, UnsupportedGraphError, dense_strides
from lowering import Schedule, UnsupportedScheduleError, expression

COORD = '!cute.coord<"(?,?)">'
INDENT = "      "


@dataclass(frozen=True)
class ScanSchedule:
    arch: str = "sm_90"
    threads: int = 128
    elements_per_thread: int = 4

    def __post_init__(self) -> None:
        Schedule(arch=self.arch, threads=self.threads)
        if type(self.elements_per_thread) is not int or not (
            1 <= self.elements_per_thread <= 16
        ):
            raise UnsupportedScheduleError("elements_per_thread must be 1 to 16")

    @property
    def tile(self) -> int:
        return self.threads * self.elements_per_thread

    @property
    def warps(self) -> int:
        return self.threads // 32


@dataclass(frozen=True)
class ScanLowered:
    schedule: ScanSchedule
    name: str
    mlir: str
    grid: tuple[int, int, int]
    output_shapes: tuple[Shape, ...]
    shared_memory_bytes: int


def axis_layout(shape: Shape, strides: Strides, axis: int) -> str:
    """``((1, batch...), n):((0, batch strides...), axis stride)``.

    The leading extent-1 mode keeps the batch mode a tuple for rank-1 arrays;
    every memref of one kernel uses the same batch shape, so the flattened row
    coordinate names the same batch position in each of them.
    """
    batch = [
        (extent, stride)
        for k, (extent, stride) in enumerate(zip(shape, strides))
        if k != axis
    ]
    extents = ",".join(str(extent) for extent in (1, *(extent for extent, _ in batch)))
    offsets = ",".join(str(stride) for stride in (0, *(stride for _, stride in batch)))
    return f"(({extents}),{shape[axis]}):(({offsets}),{strides[axis]})"


def gmem(layout: str) -> str:
    return f'!cute.memref<f32, gmem, align<4>, "{layout}">'


def load(memory: str, kind: str, coord: str, result: str) -> str:
    return f"{result} = cute.memref.load({memory}, {coord}) : ({kind}, {COORD}) -> f32"


def store(memory: str, kind: str, coord: str, value: str) -> str:
    return (
        f"cute.memref.store({memory}, {coord}, {value}) : ({kind}, {COORD}, f32) -> ()"
    )


def validate_combine(graph: Graph) -> int:
    """The combine is traced on scalars: 2k inputs, k outputs, elementwise nodes."""
    leaves = len(graph.outputs)
    if len(graph.inputs) != 2 * leaves:
        raise UnsupportedGraphError(
            "a combine takes two operands with the output structure"
        )
    values = (*graph.inputs, *graph.outputs, *(node.output for node in graph.nodes))
    if any(value.shape != () for value in values):
        raise UnsupportedGraphError("the combine must be traced on scalars")
    if any(node.operation in ("ReduceSum", "Transpose") for node in graph.nodes):
        raise UnsupportedGraphError("the combine must be elementwise")
    return leaves


class Combine:
    """Emit the combine graph on SSA scalars with names unique per application."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.leaves = validate_combine(graph)
        self.applications = 0

    def constants(self) -> list[str]:
        import struct

        lines = []
        for i, (_, value) in enumerate(self.graph.constants):
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            lines.append(f"%k{i} = arith.constant 0x{bits:08X} : f32")
        return lines

    def apply(self, left: list[str], right: list[str]) -> tuple[list[str], list[str]]:
        names = {
            value.name: ssa for value, ssa in zip(self.graph.inputs, (*left, *right))
        }
        names.update(
            (name, f"%k{i}") for i, (name, _) in enumerate(self.graph.constants)
        )
        self.applications += 1
        lines = []
        for i, node in enumerate(self.graph.nodes):
            result = expression(node, names)
            if node.operation == "Broadcast":
                names[node.output.name] = result
                continue
            names[node.output.name] = f"%c{self.applications}_{i}"
            lines.append(f"{names[node.output.name]} = {result}")
        return lines, [names[output.name] for output in self.graph.outputs]


def kernel(name: str, params: list[str], threads: int, body: list[str]) -> str:
    lines = [
        "module attributes {gpu.container_module} {",
        "  gpu.module @kernels {",
        f"    cuda.kernel @{name}({', '.join(params)}) attributes {{cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: {threads}, 1, 1>}} {{",
        *(INDENT + line for line in body),
        INDENT + "return",
        "    }",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def prologue(shape: Shape, axis: int, tiles: int, schedule: ScanSchedule) -> list[str]:
    return [
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


def warp_scan(combine: Combine, values: list[str]) -> tuple[list[str], list[str]]:
    """Inclusive scan of one value per lane; lanes below the shift keep their own value."""
    lines = []
    for shift in (1, 2, 4, 8, 16):
        lines.append(f"%shift{shift} = arith.constant {shift} : i32")
        shifted = []
        for i, value in enumerate(values):
            shifted.append(f"%up{i}_{shift}")
            lines.append(
                f"{shifted[i]} = nvvm.shfl.sync up %mask, {value}, %shift{shift}, %clamp : f32 -> f32"
            )
        lines.append(f"%has{shift} = arith.cmpi uge, %lane, %shift{shift} : i32")
        combined, results = combine.apply(shifted, values)
        lines.extend(combined)
        for i, value in enumerate(values):
            lines.append(
                f"%warp{i}_{shift} = arith.select %has{shift}, {results[i]}, {value} : f32"
            )
        values = [f"%warp{i}_{shift}" for i in range(len(values))]
    return lines, values


def block_prefix(
    combine: Combine, schedule: ScanSchedule, totals: list[str], exclusive: list[str]
) -> tuple[list[str], list[str], str]:
    """Exclusive prefix of every thread from the warp totals staged in shared memory."""
    leaves = combine.leaves
    kind = (
        f'!cute.memref<f32, smem, align<4>, "({schedule.warps},{leaves}):({leaves},1)">'
    )
    lines = [
        f"%shared = cute.memref.alloca() : {kind}",
        "%lane_last = arith.constant 31 : i32",
    ]
    lines.append("%is_last = arith.cmpi eq, %lane, %lane_last : i32")
    for i in range(leaves):
        lines.append(f"%slot{i} = arith.constant {i} : i32")
    lines.append("scf.if %is_last {")
    for i in range(leaves):
        lines.append(
            f"  %stage{i} = cute.make_coord(%warp, %slot{i}) : (i32, i32) -> {COORD}"
        )
        lines.append("  " + store("%shared", kind, f"%stage{i}", totals[i]))
    lines.extend(["}", "nvvm.barrier"])
    for i in range(leaves):
        lines.append(
            f"%first_coord{i} = cute.make_coord(%zero, %slot{i}) : (i32, i32) -> {COORD}"
        )
        lines.append(load("%shared", kind, f"%first_coord{i}", f"%first{i}"))
    iterated = ", ".join(f"%fold{i} = %first{i}" for i in range(leaves))
    types = ", ".join("f32" for _ in range(leaves))
    lines.append(
        f"%carry:{leaves} = scf.for %j = %one to %warp step %one iter_args({iterated}) -> ({types}) : i32 {{"
    )
    for i in range(leaves):
        lines.append(
            f"  %next_coord{i} = cute.make_coord(%j, %slot{i}) : (i32, i32) -> {COORD}"
        )
        lines.append("  " + load("%shared", kind, f"%next_coord{i}", f"%next{i}"))
    folded, results = combine.apply(
        [f"%fold{i}" for i in range(leaves)], [f"%next{i}" for i in range(leaves)]
    )
    lines.extend("  " + line for line in folded)
    lines.append(f"  scf.yield {', '.join(results)} : {types}")
    lines.append("}")
    lines.append("%has_warp = arith.cmpi uge, %warp, %one : i32")
    carry = [f"%carry#{i}" for i in range(leaves)]
    both, results = combine.apply(carry, exclusive)
    lines.extend(both)
    prefix = []
    for i in range(leaves):
        lines.append(
            f"%in_warp{i} = arith.select %has_lane, {results[i]}, {carry[i]} : f32"
        )
        lines.append(
            f"%prefix{i} = arith.select %has_warp, %in_warp{i}, {exclusive[i]} : f32"
        )
        prefix.append(f"%prefix{i}")
    lines.append("%has_prefix = arith.ori %has_warp, %has_lane : i1")
    return lines, prefix, "%has_prefix"


def lower_tile_scan(
    combine_graph: Graph,
    profiles: tuple[Profile, ...],
    axis: int,
    schedule: ScanSchedule,
) -> ScanLowered:
    """One kernel per (row, tile): the tile-local inclusive scan and the tile aggregate."""
    combine = Combine(combine_graph)
    leaves = combine.leaves
    shape = profiles[0][0]
    n = shape[axis]
    rows = prod(shape) // n
    tiles = (n + schedule.tile - 1) // schedule.tile
    inputs = [gmem(axis_layout(shape, strides, axis)) for _, strides in profiles]
    local = gmem(axis_layout(shape, dense_strides(shape), axis))
    aggregate = gmem(f"({rows},{tiles}):({tiles},1)")
    kinds = [*inputs, *[local] * leaves, *[aggregate] * leaves]
    params = [f"%arg{i}: {kind}" for i, kind in enumerate(kinds)]
    body = [
        *prologue(shape, axis, tiles, schedule),
        f"%last = arith.constant {n - 1} : i32",
        "%warp_size = arith.constant 32 : i32",
        "%lane = arith.remui %thread, %warp_size : i32",
        "%warp = arith.divui %thread, %warp_size : i32",
        "%mask = arith.constant -1 : i32",
        "%clamp = arith.constant 0 : i32",
        *combine.constants(),
    ]
    chunks: list[list[str]] = []
    for e in range(schedule.elements_per_thread):
        body.append(f"%offset{e} = arith.constant {e} : i32")
        body.append(f"%position{e} = arith.addi %chunk_base, %offset{e} : i32")
        body.append(f"%clamped{e} = arith.minsi %position{e}, %last : i32")
        body.append(
            f"%coord{e} = cute.make_coord(%row, %clamped{e}) : (i32, i32) -> {COORD}"
        )
        loaded = [f"%load{i}_{e}" for i in range(leaves)]
        body.extend(
            load(f"%arg{i}", inputs[i], f"%coord{e}", loaded[i]) for i in range(leaves)
        )
        if e == 0:
            chunks.append(loaded)
            continue
        lines, results = combine.apply(chunks[-1], loaded)
        body.extend(lines)
        chunks.append(results)
    lines, totals = warp_scan(combine, chunks[-1])
    body.extend(lines)
    exclusive = [f"%exclusive{i}" for i in range(leaves)]
    body.extend(
        f"{exclusive[i]} = nvvm.shfl.sync up %mask, {totals[i]}, %one, %clamp : f32 -> f32"
        for i in range(leaves)
    )
    body.append("%has_lane = arith.cmpi uge, %lane, %one : i32")
    if schedule.warps > 1:
        lines, prefix, has_prefix = block_prefix(combine, schedule, totals, exclusive)
        body.extend(lines)
    else:
        prefix, has_prefix = exclusive, "%has_lane"
    final: list[str] = []
    for e, chunk in enumerate(chunks):
        lines, results = combine.apply(prefix, chunk)
        body.extend(lines)
        final = [f"%out{i}_{e}" for i in range(leaves)]
        body.extend(
            f"{final[i]} = arith.select {has_prefix}, {results[i]}, {chunk[i]} : f32"
            for i in range(leaves)
        )
        body.append(f"%valid{e} = arith.cmpi ult, %position{e}, %length : i32")
        body.append(f"scf.if %valid{e} {{")
        body.append(
            f"  %out_coord{e} = cute.make_coord(%row, %position{e}) : (i32, i32) -> {COORD}"
        )
        body.extend(
            "  " + store(f"%arg{leaves + i}", local, f"%out_coord{e}", final[i])
            for i in range(leaves)
        )
        body.append("}")
    body.append(f"%thread_last = arith.constant {schedule.threads - 1} : i32")
    body.append("%is_tail = arith.cmpi eq, %thread, %thread_last : i32")
    body.append("scf.if %is_tail {")
    body.append(
        f"  %aggregate_coord = cute.make_coord(%row, %tile) : (i32, i32) -> {COORD}"
    )
    body.extend(
        "  " + store(f"%arg{2 * leaves + i}", aggregate, "%aggregate_coord", final[i])
        for i in range(leaves)
    )
    body.append("}")
    return ScanLowered(
        schedule,
        "tiki_scan",
        kernel("tiki_scan", params, schedule.threads, body),
        (tiles * rows * schedule.threads, 1, 1),
        (*[shape] * leaves, *[(rows, tiles)] * leaves),
        4 * schedule.warps * leaves if schedule.warps > 1 else 0,
    )


def lower_apply(
    combine_graph: Graph, shape: Shape, axis: int, tiles: int, schedule: ScanSchedule
) -> ScanLowered:
    """Fold the exclusive tile carry into every position of the tile-local scan."""
    combine = Combine(combine_graph)
    leaves = combine.leaves
    n = shape[axis]
    rows = prod(shape) // n
    carry = gmem(f"({rows},{tiles}):({tiles},1)")
    dense = gmem(axis_layout(shape, dense_strides(shape), axis))
    kinds = [*[carry] * leaves, *[dense] * (2 * leaves)]
    params = [f"%arg{i}: {kind}" for i, kind in enumerate(kinds)]
    body = [
        *prologue(shape, axis, tiles, schedule),
        *combine.constants(),
        "%has_prefix = arith.cmpi uge, %tile, %one : i32",
        "%previous = arith.subi %tile, %one : i32",
        "%previous_clamped = arith.maxsi %previous, %zero : i32",
        f"%carry_coord = cute.make_coord(%row, %previous_clamped) : (i32, i32) -> {COORD}",
    ]
    carries = [f"%carry{i}" for i in range(leaves)]
    body.extend(
        load(f"%arg{i}", carry, "%carry_coord", carries[i]) for i in range(leaves)
    )
    for e in range(schedule.elements_per_thread):
        body.append(f"%offset{e} = arith.constant {e} : i32")
        body.append(f"%position{e} = arith.addi %chunk_base, %offset{e} : i32")
        body.append(f"%valid{e} = arith.cmpi ult, %position{e}, %length : i32")
        body.append(f"scf.if %valid{e} {{")
        body.append(
            f"  %coord{e} = cute.make_coord(%row, %position{e}) : (i32, i32) -> {COORD}"
        )
        values = [f"%value{i}_{e}" for i in range(leaves)]
        body.extend(
            "  " + load(f"%arg{leaves + i}", dense, f"%coord{e}", values[i])
            for i in range(leaves)
        )
        lines, results = combine.apply(carries, values)
        body.extend("  " + line for line in lines)
        for i in range(leaves):
            body.append(
                f"  %result{i}_{e} = arith.select %has_prefix, {results[i]}, {values[i]} : f32"
            )
            body.append(
                "  "
                + store(f"%arg{2 * leaves + i}", dense, f"%coord{e}", f"%result{i}_{e}")
            )
        body.append("}")
    return ScanLowered(
        schedule,
        "tiki_scan_apply",
        kernel("tiki_scan_apply", params, schedule.threads, body),
        (tiles * rows * schedule.threads, 1, 1),
        tuple([shape] * leaves),
        0,
    )
