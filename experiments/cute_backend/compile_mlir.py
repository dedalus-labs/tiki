"""Compile a serialized CuTe MLIR artifact without importing its source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cutlass.compiler as cutlass_compiler


def cute_pipeline(cubin_prefix: Path, *, keep_ptx: bool = False) -> str:
    path = str(cubin_prefix)
    if "'" in path:
        raise ValueError("cubin output path cannot contain a single quote")
    ptx = f"dump-ptx-path='{path}' " if keep_ptx else ""
    return (
        "cute-to-nvvm{ "
        "check-inline-asm=false "
        "cubin-format=bin "
        f"dump-cubin-path='{path}' "
        f"{ptx}"
        "enable-cuda-dialect=true "
        "cuda-dialect-external-module=true "
        "}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cubin-output", type=Path, required=True)
    parser.add_argument("--textual", action="store_true")
    args = parser.parse_args()

    data = args.input.read_bytes()
    if args.textual:
        artifact = cutlass_compiler.PreCompiledMlirArtifact.from_textual_form(data)
    else:
        artifact = cutlass_compiler.deserialize_compilation_artifact(data)
    if not isinstance(artifact, cutlass_compiler.PreCompiledMlirArtifact):
        raise TypeError(f"expected precompiled MLIR, got {artifact.type_name}")

    compiler = cutlass_compiler.CuteCompiler()
    compiler.set_device_target(args.arch)
    cubin_prefix = args.cubin_output.with_name(args.cubin_output.name + ".dump")
    compiler.set_pipeline(
        cutlass_compiler.ArtifactType.PreCompiledMlir,
        cute_pipeline(cubin_prefix),
    )
    if args.textual:
        compiler.set_abi(cutlass_compiler.Abi.Tbd)
    compiled = compiler.compile_to(
        artifact,
        cutlass_compiler.ArtifactType.Object,
    )
    data = compiled.get_data()
    args.output.write_bytes(data)
    cubin_paths = tuple(cubin_prefix.parent.glob(cubin_prefix.name + ".*.cubin"))
    if len(cubin_paths) != 1:
        raise RuntimeError(f"expected one cubin, got {cubin_paths}")
    cubin_paths[0].replace(args.cubin_output)
    cubin = args.cubin_output.read_bytes()
    result = {
        "arch": args.arch,
        "input": "textual" if args.textual else "serialized",
        "object_bytes": len(data),
        "cubin_bytes": len(cubin),
        "functions": [metadata.symbol_name for metadata in compiled.metadata],
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
