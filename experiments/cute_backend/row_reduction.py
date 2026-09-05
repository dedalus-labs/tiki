"""Lower a sum and its elementwise producer/consumer to cooperative rows."""

import struct

from graph import Graph, Node, Shape, UnsupportedGraphError, Value
from lowering import Lowered, RowSchedule, expression, memref


def scalar_code(
    graph: Graph, target: str, coordinates: dict[Shape, str], known: dict[str, str]
) -> tuple[list[str], str]:
    nodes = {node.output.name: node for node in graph.nodes}
    inputs = {value.name: (i, value) for i, value in enumerate(graph.inputs)}
    constants = dict(graph.constants)
    names = dict(known)
    lines = []

    def emit(name: str) -> str:
        if name in names:
            return names[name]
        result = f"%v{len(names)}"
        if name in inputs:
            index, value = inputs[name]
            coord = coordinates[value.shape]
            rhs = f'cute.memref.load(%arg{index}, {coord}) : ({memref(value)}, !cute.coord<"?">) -> f32'
        elif name in constants:
            bits = struct.unpack("<I", struct.pack("<f", constants[name]))[0]
            rhs = f"arith.constant 0x{bits:08X} : f32"
        else:
            node = nodes[name]
            for operand in node.inputs:
                emit(operand)
            result = f"%v{len(names)}"
            rhs = expression(node, names)
            if node.operation == "Broadcast":
                names[name] = rhs
                return rhs
        names[name] = result
        lines.append(f"{result} = {rhs}")
        return result

    result = emit(target)
    return lines, result


def warp_sum(value: str, prefix: str) -> tuple[list[str], str]:
    lines = []
    for offset in (16, 8, 4, 2, 1):
        lines.append(f"%{prefix}offset{offset} = arith.constant {offset} : i32")
        lines.append(
            f"%{prefix}shuffle{offset} = nvvm.shfl.sync bfly %mask, {value}, %{prefix}offset{offset}, %clamp : f32 -> f32"
        )
        lines.append(
            f"%{prefix}sum{offset} = arith.addf {value}, %{prefix}shuffle{offset} : f32"
        )
        value = f"%{prefix}sum{offset}"
    return lines, value


def validate_row(graph: Graph) -> Node | None:
    if len(graph.shape) != 2 or graph.shape[1] == 0:
        raise UnsupportedGraphError(
            "row schedule requires a nonzero width and a 2D output"
        )
    rows, cols = graph.shape
    values = (*graph.inputs, *(node.output for node in graph.nodes))
    if any(
        value.shape not in ((), (cols,), (rows, 1), (rows, cols)) for value in values
    ):
        raise UnsupportedGraphError("unsupported row broadcast shape")
    reductions = [node for node in graph.nodes if node.operation == "ReduceSum"]
    if not reductions and cols == 1:
        return None
    if len(reductions) != 1 or any(
        node.operation == "Transpose" for node in graph.nodes
    ):
        raise UnsupportedGraphError("row schedule requires exactly one sum reduction")
    reduction = reductions[0]
    shapes = {value.name: value.shape for value in values}
    if shapes[reduction.inputs[0]] != graph.shape or reduction.output.shape != (
        rows,
        1,
    ):
        raise UnsupportedGraphError("sum must reduce each full output row")
    return reduction


def coordinates(graph: Graph) -> tuple[list[str], dict[Shape, str]]:
    rows, cols = graph.shape
    lines = [
        "%flat = arith.addi %row_offset, %column : i32",
        '%full_coord = cute.make_coord(%flat) : (i32) -> !cute.coord<"?">',
        '%column_coord = cute.make_coord(%column) : (i32) -> !cute.coord<"?">',
    ]
    return lines, {
        (): "%scalar_coord",
        (cols,): "%column_coord",
        (rows, 1): "%row_coord",
        (rows, cols): "%full_coord",
    }


def reduction_loop(graph: Graph, reduction: Node) -> list[str]:
    coords, mapping = coordinates(graph)
    code, result = scalar_code(graph, reduction.inputs[0], mapping, {})
    return [
        "%partial = scf.for %column = %local_thread to %width step %row_threads iter_args(%sum = %zero_f) -> (f32) : i32 {",
        "  %next = scf.if %row_valid -> (f32) {",
        *("    " + line for line in (*coords, *code)),
        f"    %added = arith.addf %sum, {result} : f32",
        "    scf.yield %added : f32",
        "  } else {",
        "    scf.yield %sum : f32",
        "  }",
        "  scf.yield %next : f32",
        "}",
    ]


def block_sum(schedule: RowSchedule, value: str) -> tuple[list[str], str]:
    warps = schedule.threads_per_row // 32
    if warps == 1:
        return [], value
    shared = f'!cute.memref<f32, smem, align<4>, "({schedule.rows_per_block},{warps}):({warps},1)">'
    lines = [
        f"%shared = cute.memref.alloca() : {shared}",
        "%warp = arith.divui %local_thread, %warp_size : i32",
        "%leader = arith.cmpi eq, %lane, %zero : i32",
        "scf.if %leader {",
        '  %coord = cute.make_coord(%local_row, %warp) : (i32, i32) -> !cute.coord<"(?,?)">',
        f'  cute.memref.store(%shared, %coord, {value}) : ({shared}, !cute.coord<"(?,?)">, f32) -> ()',
        "}",
        "nvvm.barrier",
        f"%warps = arith.constant {warps} : i32",
        "%active_lane = arith.cmpi ult, %lane, %warps : i32",
        "%summary = scf.if %active_lane -> (f32) {",
        '  %coord = cute.make_coord(%local_row, %lane) : (i32, i32) -> !cute.coord<"(?,?)">',
        f'  %loaded = cute.memref.load(%shared, %coord) : ({shared}, !cute.coord<"(?,?)">) -> f32',
        "  scf.yield %loaded : f32",
        "} else {",
        "  scf.yield %zero_f : f32",
        "}",
    ]
    shuffles, total = warp_sum("%summary", "block")
    return [*lines, *shuffles], total


def output_loop(graph: Graph, known: dict[str, str]) -> list[str]:
    coords, mapping = coordinates(graph)
    code, result = scalar_code(graph, graph.output, mapping, known)
    output = Value(graph.output, graph.shape)
    return [
        "scf.if %row_valid {",
        "  scf.for %column = %local_thread to %width step %row_threads : i32 {",
        *("    " + line for line in (*coords, *code)),
        f'    cute.memref.store(%arg{len(graph.inputs)}, %full_coord, {result}) : ({memref(output)}, !cute.coord<"?">, f32) -> ()',
        "  }",
        "}",
    ]


def lower_row(graph: Graph, schedule: RowSchedule) -> Lowered:
    reduction = validate_row(graph)
    if graph.shape[0] == 0:
        return Lowered(graph, schedule, "module {}\n")
    params = ", ".join(
        f"%arg{i}: {memref(value)}"
        for i, value in enumerate((*graph.inputs, Value(graph.output, graph.shape)))
    )
    prologue = [
        "%thread = nvvm.read.ptx.sreg.tid.x : i32",
        "%block = nvvm.read.ptx.sreg.ctaid.x : i32",
        f"%row_threads = arith.constant {schedule.threads_per_row} : i32",
        f"%block_rows = arith.constant {schedule.rows_per_block} : i32",
        f"%rows = arith.constant {graph.shape[0]} : i32",
        f"%width = arith.constant {graph.shape[1]} : i32",
        "%warp_size = arith.constant 32 : i32",
        "%zero = arith.constant 0 : i32",
        "%zero_f = arith.constant 0.0 : f32",
        "%mask = arith.constant -1 : i32",
        "%clamp = arith.constant 31 : i32",
        "%local_row = arith.divui %thread, %row_threads : i32",
        "%local_thread = arith.remui %thread, %row_threads : i32",
        "%lane = arith.remui %thread, %warp_size : i32",
        "%row_base = arith.muli %block, %block_rows : i32",
        "%row = arith.addi %row_base, %local_row : i32",
        "%row_offset = arith.muli %row, %width : i32",
        "%row_valid = arith.cmpi ult, %row, %rows : i32",
        '%scalar_coord = cute.make_coord(%zero) : (i32) -> !cute.coord<"?">',
        '%row_coord = cute.make_coord(%row) : (i32) -> !cute.coord<"?">',
    ]
    body = [*prologue, *cooperative_body(graph, schedule, reduction)]
    lines = [
        "module attributes {gpu.container_module} {",
        "  gpu.module @kernels {",
        f"    cuda.kernel @tiki_fused({params}) attributes {{cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: {schedule.threads}, 1, 1>}} {{",
        *("      " + line for line in body),
        "      return",
        "    }",
        "  }",
        "}",
    ]
    return Lowered(graph, schedule, "\n".join(lines) + "\n")


def cooperative_body(
    graph: Graph, schedule: RowSchedule, reduction: Node | None
) -> list[str]:
    if reduction is None:
        return output_loop(graph, {})
    warp_lines, partial = warp_sum("%partial", "warp")
    block_lines, total = block_sum(schedule, partial)
    return [
        *reduction_loop(graph, reduction),
        *warp_lines,
        *block_lines,
        *output_loop(graph, {reduction.output.name: total}),
    ]
