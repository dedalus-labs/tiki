# Python lowering validation

Validated September 10, 2026, using Python 3.14, Ruff 0.16.5, ty 0.0.79,
CuTe DSL 4.7.1, and the CUDA MLX wheel from the preceding collective-lowering
change. This refactor changes no C++ or Rust implementation.

| Check | Result |
| --- | --- |
| Ruff, formatting, ty | Compiler, entry points, tests, demos, and binding declarations pass. |
| Size limits | Largest compiler module: 238 lines. Largest function: 63 lines. Enforced limits: 300/70 lines, three control-flow levels, five arguments. |
| Local CPU suite | 72 cases: 35 passed; 37 require CUDA or launched ranks. |
| L4 mixed programs | All 11 cases pass on each of two nodes. |
| Two-rank NCCL | All 5 cases and `demo_overlap.py` pass on both nodes. |
| Artifact and derivative contracts | Both cases pass on each node. |
| L4 scan suite | All 18 cases pass, including forward/reverse evaluation, derivatives, strided inputs, and training. |
| Source identity | SHA-256 digests of all 26 compiler/entry-point files match the validated files on both nodes. |

The native tests cover all-reduce sum/min/max, all-gather, sum-scatter,
communicator identity, and ordering across streams. These were networked L4
nodes running NCCL. This validation does not establish a performance improvement
or exercise CuTe device collectives.

## Reproduced failures

The old boundary accepted a profile whose shape disagreed with the export, a
duplicate input header, and unmodeled arguments on stateless arithmetic.
The new tests first reproduced each acceptance and now require a lowering error.
Negative type fixtures reject string operation identities, nonnumeric thread
counts, a matrix stage carrying a communicator index, and a collective without
a group index.

A scan selected for `sm_89` created `sm_90` derivative kernels. Its GPU test
failed with `schedule requires sm_90`; derivative schedules now inherit the
forward target.

Forward-mode differentiation entered a raw `CustomKernel` before reaching the
registered rule. A standalone probe established that stopping traversal at the
private forward boundary preserves both registered derivatives. The compiled
call uses that boundary; low-level launch still rejects raw-kernel differentiation.

The previous MLX GPU tree comparison disagreed with a sequential float64
recurrence on real and complex affine derivatives. In the coupled complex case,
Tiki's maximum error was about `1.3e-6`, while the tree comparison differed by up
to `6.04`. Affine derivative sweeps now use independent float64 oracles with
`rtol=atol=1e-4`.

## Emission preservation

Full MLIR text and launch metadata were recorded before refactoring. All six
cases remain byte-identical, in addition to the execution checks above.

| Case | MLIR SHA-256 |
| --- | --- |
| Elementwise, 7 values | `7b158b6663af4ddf9c38d8c8d177296838f4f682216e4cda92b0f536f0ab0e2e` |
| Strided elementwise, 3×7 | `9693edcab631872270be7f2c4c2b79cc1ec89c00f7215269940caa686280bfa1` |
| Row reduction, 3×65 | `bf710d5fabb0c664519de50e720493ace0733264cd9491cea392a817d7c872ef` |
| Transpose, 33×65 | `8d5df3cc705a04a417ebd339ca6d121fe2a4f8ab756f1633dd90c9229d3b44cd` |
| Scan tile, 2×257 | `4f2347bd1ec4b2b07af8c156e5f85ced1413a1802fee97611271760113c35607` |
| Scan carry, 2×257 | `30ab7b6b77e1d011ed85a815031d5bbeeb3ee6714df600b4e385ab65ccd15dc2` |

Commands are in the [compiler contract](tiki_compiler/README.md).
