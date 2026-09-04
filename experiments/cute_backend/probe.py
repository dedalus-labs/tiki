"""Prove that serialized CuTe MLIR is a standalone compiler input."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cutlass
import cutlass.compiler as cutlass_compiler
from cutlass import cute

THREADS = 128
CUDA_KERNEL = re.compile(r"cuda\.kernel\s+@([A-Za-z0-9_.$]+)")


@cute.kernel
def add_kernel(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * THREADS + thread
    if index < a.shape[0]:
        output[index] = a[index] + b[index]


@cute.jit
def add(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor):
    blocks = cute.ceil_div(a.shape[0], THREADS)
    add_kernel(a, b, output).launch(
        grid=(blocks, 1, 1),
        block=(THREADS, 1, 1),
    )


def fake_vector(size: int):
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (size,),
        stride_order=(0,),
        assumed_align=16,
    )


def compile_reference(size: int, arch: str):
    tensors = tuple(fake_vector(size) for _ in range(3))
    artifact = cute.compile_to(
        cutlass_compiler.ArtifactType.PreCompiledMlir,
        add,
        *tensors,
        options=f"--gpu-arch {arch}",
    )
    return artifact


def extract_kernel_name(mlir: str) -> str:
    kernel_names = tuple(CUDA_KERNEL.findall(mlir))
    if len(kernel_names) != 1:
        raise RuntimeError(f"expected one CUDA kernel in MLIR, got {kernel_names}")
    return kernel_names[0]


def compile_reference_cubin(size: int, arch: str) -> tuple[bytes, str]:
    precompiled = compile_reference(size, arch)
    serialized = cutlass_compiler.serialize_compilation_artifact(precompiled)
    with tempfile.TemporaryDirectory(prefix="tiki-cute-") as directory:
        output = Path(directory)
        mlir_path = output / "add.precompiled.mlirbin"
        object_path = output / "add.o"
        cubin_path = output / "add.cubin"
        mlir_path.write_bytes(serialized)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("compile_mlir.py")),
                "--arch",
                arch,
                "--input",
                str(mlir_path),
                "--output",
                str(object_path),
                "--cubin-output",
                str(cubin_path),
            ],
            check=True,
        )
        return cubin_path.read_bytes(), extract_kernel_name(str(precompiled))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("size must be positive")

    args.output.mkdir(parents=True, exist_ok=False)
    precompiled = compile_reference(size=args.size, arch=args.arch)
    textual_mlir = str(precompiled)
    kernel_name = extract_kernel_name(textual_mlir)
    (args.output / "add.mlir").write_text(textual_mlir + "\n")
    serialized = cutlass_compiler.serialize_compilation_artifact(precompiled)
    mlir_path = args.output / "add.precompiled.mlirbin"
    object_path = args.output / "add.o"
    cubin_path = args.output / "add.cubin"
    mlir_path.write_bytes(serialized)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("compile_mlir.py")),
            "--arch",
            args.arch,
            "--input",
            str(mlir_path),
            "--output",
            str(object_path),
            "--cubin-output",
            str(cubin_path),
        ],
        check=True,
    )
    object_data = object_path.read_bytes()
    compiled_result = json.loads(object_path.with_suffix(".json").read_text())

    direct_object_path = args.output / "add-direct.o"
    direct_cubin_path = args.output / "add-direct.cubin"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("compile_mlir.py")),
            "--arch",
            args.arch,
            "--input",
            str(args.output / "add.mlir"),
            "--output",
            str(direct_object_path),
            "--cubin-output",
            str(direct_cubin_path),
            "--textual",
        ],
        check=True,
    )
    direct_object_data = direct_object_path.read_bytes()
    cubin_data = cubin_path.read_bytes()
    direct_cubin_data = direct_cubin_path.read_bytes()

    result = {
        "arch": args.arch,
        "size": args.size,
        "kernel": kernel_name,
        "cutlass_version": cutlass.__version__,
        "precompiled_mlir_bytes": len(serialized),
        "precompiled_mlir_sha256": sha256(serialized),
        "textual_mlir_bytes": (args.output / "add.mlir").stat().st_size,
        "object_bytes": len(object_data),
        "object_sha256": sha256(object_data),
        "cubin_bytes": len(cubin_data),
        "cubin_sha256": sha256(cubin_data),
        "direct_mlir_object_bytes": len(direct_object_data),
        "direct_mlir_object_sha256": sha256(direct_object_data),
        "direct_mlir_cubin_bytes": len(direct_cubin_data),
        "direct_mlir_cubin_sha256": sha256(direct_cubin_data),
        "functions": compiled_result["functions"],
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
