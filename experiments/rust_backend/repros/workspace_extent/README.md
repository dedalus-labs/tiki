# CUDA workspace extent

Run `bash check.sh` to exercise the production `allocate_workspace` function with
an allocation recorder. It does not allocate device memory or require CUDA.
The recorder checks that allocation size and array extent agree, cover the
requested bytes, and reach temporary retention and address exposure exactly once.

The cases cover zero, 256-byte rounding, the signed 32-bit and 4 GiB boundaries,
the largest representable workspace shape, and `SIZE_MAX`. Requests beyond the
two-axis byte shape are rejected before allocation. An optional source path
allows the same fixture to qualify an earlier implementation.

The previous function narrowed the rounded byte count to `int`: requesting
4 GiB + 256 bytes allocated only 256 bytes. The caller still gave the original
workspace size to its CUDA library. The two-axis shape preserves wide byte
counts up to 512 GiB minus 256 bytes without overflowing either dimension.

Set `WORKSPACE_TEST_FLAGS=-fsanitize=address,undefined` for host sanitizers. CUDA
library execution and real large-allocation tests remain separate checks.
