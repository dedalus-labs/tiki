"""Emit tile-local scans and application of recursively scanned tile carries."""

from math import prod

from .graph import Graph, Profile, Shape, dense_strides
from .scan_primitives import (
    COORD,
    Combine,
    axis_layout,
    block_prefix,
    gmem,
    kernel,
    load,
    prologue,
    store,
    warp_scan,
)
from .scan_schedule import ScanLowered, ScanSchedule


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
    length = shape[axis]
    rows = prod(shape) // length
    tiles = (length + schedule.tile - 1) // schedule.tile
    inputs = [gmem(axis_layout(shape, strides, axis)) for _, strides in profiles]
    local = gmem(axis_layout(shape, dense_strides(shape), axis))
    aggregate = gmem(f"({rows},{tiles}):({tiles},1)")
    kinds = [*inputs, *[local] * leaves, *[aggregate] * leaves]
    params = [f"%arg{index}: {kind}" for index, kind in enumerate(kinds)]
    body = [
        *prologue(shape, axis, tiles, schedule),
        f"%last = arith.constant {length - 1} : i32",
        "%warp_size = arith.constant 32 : i32",
        "%lane = arith.remui %thread, %warp_size : i32",
        "%warp = arith.divui %thread, %warp_size : i32",
        "%mask = arith.constant -1 : i32",
        "%clamp = arith.constant 0 : i32",
        *combine.constants(),
    ]
    lines, chunks = scan_chunks(combine, inputs, schedule)
    body.extend(lines)
    lines, totals = warp_scan(combine, chunks[-1])
    body.extend(lines)
    exclusive = [f"%exclusive{leaf}" for leaf in range(leaves)]
    body.extend(
        f"{exclusive[leaf]} = nvvm.shfl.sync up %mask, {totals[leaf]}, %one, %clamp : f32 -> f32"
        for leaf in range(leaves)
    )
    body.append("%has_lane = arith.cmpi uge, %lane, %one : i32")
    if schedule.warps > 1:
        lines, prefix, has_prefix = block_prefix(combine, schedule, totals, exclusive)
        body.extend(lines)
    else:
        prefix, has_prefix = exclusive, "%has_lane"
    lines, final = store_chunks(combine, chunks, prefix, has_prefix, local)
    body.extend(lines)
    body.append(f"%thread_last = arith.constant {schedule.threads - 1} : i32")
    body.append("%is_tail = arith.cmpi eq, %thread, %thread_last : i32")
    body.append("scf.if %is_tail {")
    body.append(f"  %aggregate_coord = cute.make_coord(%row, %tile) : (i32, i32) -> {COORD}")
    body.extend(
        "  " + store(f"%arg{2 * leaves + leaf}", aggregate, "%aggregate_coord", final[leaf])
        for leaf in range(leaves)
    )
    body.append("}")
    lowered = ScanLowered(
        schedule=schedule,
        name="tiki_scan",
        mlir=kernel("tiki_scan", params, schedule.threads, body),
        grid=(tiles * rows * schedule.threads, 1, 1),
        output_shapes=(*[shape] * leaves, *[(rows, tiles)] * leaves),
        shared_memory_bytes=4 * schedule.warps * leaves if schedule.warps > 1 else 0,
    )
    return lowered


def lower_apply(
    combine_graph: Graph, shape: Shape, axis: int, tiles: int, schedule: ScanSchedule
) -> ScanLowered:
    """Fold the exclusive tile carry into every position of the tile-local scan."""
    combine = Combine(combine_graph)
    leaves = combine.leaves
    length = shape[axis]
    rows = prod(shape) // length
    carry = gmem(f"({rows},{tiles}):({tiles},1)")
    dense = gmem(axis_layout(shape, dense_strides(shape), axis))
    kinds = [*[carry] * leaves, *[dense] * (2 * leaves)]
    params = [f"%arg{index}: {kind}" for index, kind in enumerate(kinds)]
    body = [
        *prologue(shape, axis, tiles, schedule),
        *combine.constants(),
        "%has_prefix = arith.cmpi uge, %tile, %one : i32",
        "%previous = arith.subi %tile, %one : i32",
        "%previous_clamped = arith.maxsi %previous, %zero : i32",
        f"%carry_coord = cute.make_coord(%row, %previous_clamped) : (i32, i32) -> {COORD}",
    ]
    carries = [f"%carry{leaf}" for leaf in range(leaves)]
    body.extend(load(f"%arg{leaf}", carry, "%carry_coord", carries[leaf]) for leaf in range(leaves))
    body.extend(apply_elements(combine, schedule, dense, carries))
    lowered = ScanLowered(
        schedule=schedule,
        name="tiki_scan_apply",
        mlir=kernel("tiki_scan_apply", params, schedule.threads, body),
        grid=(tiles * rows * schedule.threads, 1, 1),
        output_shapes=tuple([shape] * leaves),
        shared_memory_bytes=0,
    )
    return lowered


def scan_chunks(
    combine: Combine, inputs: list[str], schedule: ScanSchedule
) -> tuple[list[str], list[list[str]]]:
    body: list[str] = []
    leaves = combine.leaves
    chunks: list[list[str]] = []
    for element in range(schedule.elements_per_thread):
        body.append(f"%offset{element} = arith.constant {element} : i32")
        body.append(f"%position{element} = arith.addi %chunk_base, %offset{element} : i32")
        body.append(f"%clamped{element} = arith.minsi %position{element}, %last : i32")
        body.append(
            f"%coord{element} = cute.make_coord(%row, %clamped{element}) : (i32, i32) -> {COORD}"
        )
        loaded = [f"%load{leaf}_{element}" for leaf in range(leaves)]
        body.extend(
            load(f"%arg{leaf}", inputs[leaf], f"%coord{element}", loaded[leaf])
            for leaf in range(leaves)
        )
        if element == 0:
            chunks.append(loaded)
            continue
        lines, results = combine.apply(chunks[-1], loaded)
        body.extend(lines)
        chunks.append(results)
    scanned = body, chunks
    return scanned


def store_chunks(
    combine: Combine, chunks: list[list[str]], prefix: list[str], has_prefix: str, local: str
) -> tuple[list[str], list[str]]:
    body: list[str] = []
    leaves = combine.leaves
    final: list[str] = []
    for element, chunk in enumerate(chunks):
        lines, results = combine.apply(prefix, chunk)
        body.extend(lines)
        final = [f"%out{leaf}_{element}" for leaf in range(leaves)]
        body.extend(
            f"{final[leaf]} = arith.select {has_prefix}, {results[leaf]}, {chunk[leaf]} : f32"
            for leaf in range(leaves)
        )
        body.append(f"%valid{element} = arith.cmpi ult, %position{element}, %length : i32")
        body.append(f"scf.if %valid{element} {{")
        body.append(
            f"  %out_coord{element} = cute.make_coord(%row, "
            f"%position{element}) : (i32, i32) -> {COORD}"
        )
        body.extend(
            "  " + store(f"%arg{leaves + leaf}", local, f"%out_coord{element}", final[leaf])
            for leaf in range(leaves)
        )
        body.append("}")
    stored = body, final
    return stored


def apply_elements(
    combine: Combine, schedule: ScanSchedule, dense: str, carries: list[str]
) -> list[str]:
    body: list[str] = []
    leaves = combine.leaves
    for element in range(schedule.elements_per_thread):
        body.append(f"%offset{element} = arith.constant {element} : i32")
        body.append(f"%position{element} = arith.addi %chunk_base, %offset{element} : i32")
        body.append(f"%valid{element} = arith.cmpi ult, %position{element}, %length : i32")
        body.append(f"scf.if %valid{element} {{")
        body.append(
            f"  %coord{element} = cute.make_coord(%row, %position{element}) : (i32, i32) -> {COORD}"
        )
        values = [f"%value{leaf}_{element}" for leaf in range(leaves)]
        body.extend(
            "  " + load(f"%arg{leaves + leaf}", dense, f"%coord{element}", values[leaf])
            for leaf in range(leaves)
        )
        lines, results = combine.apply(carries, values)
        body.extend("  " + line for line in lines)
        for leaf in range(leaves):
            body.append(
                f"  %result{leaf}_{element} = arith.select %has_prefix, "
                f"{results[leaf]}, {values[leaf]} : f32"
            )
            body.append(
                "  "
                + store(
                    f"%arg{2 * leaves + leaf}",
                    dense,
                    f"%coord{element}",
                    f"%result{leaf}_{element}",
                )
            )
        body.append("}")
    return body
