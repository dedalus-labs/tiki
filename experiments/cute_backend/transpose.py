"""Lower a tiled transpose with an explicit CuTe shared-memory swizzle."""

from math import prod

from graph import Graph, UnsupportedGraphError, Value
from lowering import Lowered, TransposeSchedule, memref


def transfer(graph: Graph, shared: str, store: bool) -> list[str]:
    if store:
        row_base, col_base = "%tile_column", "%tile_row"
        rows, cols = graph.shape
        value = Value(graph.output, graph.shape)
        memory = "%arg1"
    else:
        row_base, col_base = "%tile_row", "%tile_column"
        rows, cols = graph.inputs[0].shape
        value = graph.inputs[0]
        memory = "%arg0"
    lines = [
        "%tile_r = arith.divui %item, %tile_size : i32",
        "%tile_c = arith.remui %item, %tile_size : i32",
        f"%r = arith.addi {row_base}, %tile_r : i32",
        f"%c = arith.addi {col_base}, %tile_c : i32",
        f"%nrows = arith.constant {rows} : i32",
        f"%ncols = arith.constant {cols} : i32",
        "%rvalid = arith.cmpi ult, %r, %nrows : i32",
        "%cvalid = arith.cmpi ult, %c, %ncols : i32",
        "%valid = arith.andi %rvalid, %cvalid : i1",
    ]
    lines.extend(
        [
            "scf.if %valid {",
            "  %rstart = arith.muli %r, %ncols : i32",
            "  %flat = arith.addi %rstart, %c : i32",
            '  %global = cute.make_coord(%flat) : (i32) -> !cute.coord<"?">',
        ]
    )
    coords = "%tile_c, %tile_r" if store else "%tile_r, %tile_c"
    lines.append(
        f'  %local = cute.make_coord({coords}) : (i32, i32) -> !cute.coord<"(?,?)">'
    )
    if store:
        lines.extend(
            [
                f'  %value = cute.memref.load(%shared, %local) : ({shared}, !cute.coord<"(?,?)">) -> f32',
                f'  cute.memref.store({memory}, %global, %value) : ({memref(value)}, !cute.coord<"?">, f32) -> ()',
            ]
        )
    else:
        lines.extend(
            [
                f'  %value = cute.memref.load({memory}, %global) : ({memref(value)}, !cute.coord<"?">) -> f32',
                f'  cute.memref.store(%shared, %local, %value) : ({shared}, !cute.coord<"(?,?)">, f32) -> ()',
            ]
        )
    lines.append("}")
    return lines


def validate_transpose(graph: Graph) -> None:
    if len(graph.inputs) != 1 or len(graph.shape) != 2:
        raise UnsupportedGraphError("transpose schedule requires one 2D input")
    if len(graph.nodes) != 1 or graph.nodes[0].operation != "Transpose":
        raise UnsupportedGraphError("transpose schedule requires exactly one transpose")
    if graph.inputs[0].shape != graph.shape[::-1]:
        raise UnsupportedGraphError("unsupported transpose shape")


def lower_transpose(graph: Graph, schedule: TransposeSchedule) -> Lowered:
    validate_transpose(graph)
    if prod(graph.shape) == 0:
        return Lowered(graph, schedule, "module {}\n")
    swizzle = schedule.swizzle
    layout = f"S<{swizzle.bits},{swizzle.base},{swizzle.shift}> o 0 o (32,32):(32,1)"
    shared = f'!cute.memref<f32, smem, align<128>, "{layout}">'
    output = Value(graph.output, graph.shape)
    lines = [
        "module attributes {gpu.container_module} {",
        "  gpu.module @kernels {",
        f"    cuda.kernel @tiki_fused(%arg0: {memref(graph.inputs[0])}, %arg1: {memref(output)}) attributes {{cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: {schedule.threads}, 1, 1>}} {{",
        "      %thread = nvvm.read.ptx.sreg.tid.x : i32",
        "      %block = nvvm.read.ptx.sreg.ctaid.x : i32",
        "      %tile_size = arith.constant 32 : i32",
        "      %items = arith.constant 1024 : i32",
        f"      %threads = arith.constant {schedule.threads} : i32",
        f"      %column_tiles = arith.constant {(graph.inputs[0].shape[1] + 31) // 32} : i32",
        "      %block_row = arith.divui %block, %column_tiles : i32",
        "      %block_column = arith.remui %block, %column_tiles : i32",
        "      %tile_row = arith.muli %block_row, %tile_size : i32",
        "      %tile_column = arith.muli %block_column, %tile_size : i32",
        f"      %shared = cute.memref.alloca() : {shared}",
        "      scf.for %item = %thread to %items step %threads : i32 {",
        *("        " + line for line in transfer(graph, shared, store=False)),
        "      }",
        "      nvvm.barrier",
        "      scf.for %item = %thread to %items step %threads : i32 {",
        *("        " + line for line in transfer(graph, shared, store=True)),
        "      }",
        "      return",
        "    }",
        "  }",
        "}",
    ]
    return Lowered(graph, schedule, "\n".join(lines) + "\n")
