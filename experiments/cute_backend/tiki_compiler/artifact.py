"""Compile a generated kernel through the pinned NVIDIA CuTe toolchain."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from .lowered import Lowered
from .pipeline import cute_pipeline
from .scan_schedule import ScanLowered


@dataclass(frozen=True)
class CudaBinary:
    cubin: bytes
    ptx: str


class CudaIo:
    """Own NVIDIA compilation and its temporary artifact files."""

    def compile(self, mlir: str, arch: str) -> CudaBinary:
        # The optional CUDA toolchain is loaded only at the I/O boundary.
        from cutlass import compiler

        artifact = compiler.PreCompiledMlirArtifact.from_textual_form(mlir.encode())
        backend = compiler.CuteCompiler()
        backend.set_device_target(arch)
        backend.set_abi(compiler.Abi.Tbd)
        with TemporaryDirectory(prefix="tiki-compile-") as directory:
            prefix = Path(directory) / "kernel"
            backend.set_pipeline(
                compiler.ArtifactType.PreCompiledMlir, cute_pipeline(prefix, keep_ptx=True)
            )
            backend.compile_to(artifact, compiler.ArtifactType.Object)
            compiled = CudaBinary(
                (Path(directory) / f"kernel.{arch}.cubin").read_bytes(),
                (Path(directory) / f"kernel.{arch}.ptx").read_text(),
            )
            return compiled


@lru_cache(maxsize=32)
def binary(io: CudaIo, lowered: Lowered | ScanLowered) -> CudaBinary:
    compiled = io.compile(mlir=lowered.mlir, arch=lowered.schedule.arch)
    return compiled
