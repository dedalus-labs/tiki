"""Build the standalone CuTe-to-NVVM pipeline with explicit artifact paths."""

from pathlib import Path


def cute_pipeline(cubin_prefix: Path, *, keep_ptx: bool = False) -> str:
    path = str(cubin_prefix)
    if "'" in path:
        raise ValueError("cubin output path cannot contain a single quote")
    ptx = f"dump-ptx-path='{path}' " if keep_ptx else ""
    result = (
        "cute-to-nvvm{ "
        "check-inline-asm=false "
        "cubin-format=bin "
        f"dump-cubin-path='{path}' "
        f"{ptx}"
        "enable-cuda-dialect=true "
        "cuda-dialect-external-module=true "
        "}"
    )
    return result
