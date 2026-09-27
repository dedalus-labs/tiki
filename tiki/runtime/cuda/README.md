# CUDA runtime

`tiki-cuda-runtime` owns cached device memory and the lifetime of committed
work for Tiki's Rust stack. It contains no `unsafe`: every driver call goes
through [`tiki-cuda-sys`](../cuda-sys), which hands out owned streams,
events, and device memory, and never an address or a driver handle.

The C++ CUDA backend does not use this crate. The Rust stack is built beside
it, compared against it with differential tests, and replaces it in one
cutover.

## Ownership

| Resource | Owner |
| --- | --- |
| Device memory in use | The `Allocation` a buffer holds. |
| Released device memory | The allocator's cache, per device and size class. |
| Values a batch retains and the handlers it runs | The `Batch`, dropped or run on the completion worker. |
| Signal streams, one per compute stream | Completion runtime, for the process. |

## Memory ordering

Device memory records the event of its last write and the events of its
reads since. A fill or copy into memory waits on all of them, and a read
waits on the last write. Streams therefore share memory without any
ordering from the caller, and memory the cache hands to another stream waits
for its earlier uses before it is written.

Dropping memory frees it after every recorded use, in stream order. Copies to
and from host memory finish before they return, so the driver never uses a
host slice after its borrow ends. If that wait fails, the process stops
rather than return while a copy may still run.

## Submission and retirement

A nonempty batch enters the pending map before the first fallible driver
call. The runtime records an event on the compute stream, waits on that event
from the compute stream's own signal stream, and schedules a host callback.
That callback marks one batch ready and wakes the worker. It never calls the
driver or runs a handler.

Each compute stream has an independent signal stream. A completed batch does
not imply that lower-numbered batches from other streams have completed. The
worker runs handlers and drops retained values before marking the batch
retired.

An enqueue failure returns its driver error and quarantines the affected batch
for the process lifetime. Callers waiting on retirement are notified when a
failed batch leaves the pending map. A handler that panics stops the process
instead of stranding retirement waiters. The worker is one dedicated thread
with a condition variable; no future owns a batch. Handlers must not wait for
other batches on the same worker.

## Build and test

Use Rust 1.92 or later. The crate links `libcuda` through the toolkit's link
stub; the installed driver provides it at run time.

```sh
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
cargo test --release --manifest-path tiki/runtime/cuda/Cargo.toml -- --test-threads=1
```

The GPU tests need CUDA device 0 and fail by name without one. The host
harness runs the production modules against a simulated driver, without CUDA:

```sh
rustc --edition=2024 --test tiki/runtime/cuda/tests/host.rs -o /tmp/cuda-host
/tmp/cuda-host
```
