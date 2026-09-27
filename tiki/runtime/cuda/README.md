# CUDA runtime

`tiki-cuda-runtime` owns CUDA storage and the lifetime of committed work for
Tiki's Rust stack. It contains no `unsafe`: every driver call goes through
[`tiki-cuda-sys`](../cuda-sys), which holds the generated `libcuda`
declarations and a checked adapter that hands out only integers and results.

The C++ CUDA backend does not use this crate. The Rust stack is built beside
it, compared against it with differential tests, and replaces it in one
cutover.

## Ownership

| Resource | Owner |
| --- | --- |
| Device and unified storage, allocation cache, small pool | Allocator. |
| A committed batch and its retirement | Completion worker. |
| Values a batch retains and the handlers it runs | The `Batch`, dropped or run on the worker. |
| Completion events and per-compute-stream signal streams | Completion runtime. |

## Submission and retirement

A nonempty batch enters the pending map before the first fallible driver call.
The runtime records an event on the compute stream, waits on that event from
the compute stream's own signal stream, and schedules a host callback. That
callback marks one batch ready and wakes the worker. It never calls the driver
or runs a handler.

Each compute stream has an independent signal stream. A completed batch does
not imply that lower-numbered batches from other streams have completed. The
worker runs handlers and drops retained values before marking the batch
retired.

An enqueue failure returns its driver error and quarantines the affected batch
for the process lifetime. If the callback was enqueued but popping the context
fails, the callback still owns retirement and the pop error is returned.
Callers waiting on retirement are notified when a failed batch leaves the
pending map.

The worker makes each batch's device current while running its handlers, then
restores its own context. A driver failure on the worker aborts the process
instead of stranding retirement waiters. The worker is one dedicated thread
with a condition variable; no future owns a batch. Handlers must not wait for
other batches on the same worker.

## Allocation ordering

An allocation records the stream its device storage was assigned for. Reusing
cached storage on the same device updates that assignment. Migration and
device release wait for that stream's latest commit. A blocking host export
also waits for its migration copy.

This is not per-kernel use tracking. Callers supply resource retention and
cross-stream dependencies. Events, signal streams, allocator state, and
quarantined batches live for the process.

## Build and test

Use Rust 1.92 or later. The crate links `libcuda` through the toolkit's link
stub; the installed driver provides it at run time.

```sh
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
cargo test --release --manifest-path tiki/runtime/cuda/Cargo.toml -- --include-ignored --test-threads=1
```

The ignored tests need a CUDA device. The host harness runs the production
modules against a simulated driver, without CUDA:

```sh
rustc --edition=2024 --test tiki/runtime/cuda/tests/host.rs -o /tmp/cuda-host
/tmp/cuda-host --include-ignored
```
