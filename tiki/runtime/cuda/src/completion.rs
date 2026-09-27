// Copyright © 2026 Dedalus Labs, Inc.

//! Completion runtime: one event per committed batch, retired by a worker
//! once the device signals it, plus device-side ordering for the allocator.
//!
//! Protocol:
//!
//!  1. `commit(device, stream, batch)` records an event on `stream` and keeps
//!     it as the stream's latest. When the batch retains anything, the
//!     stream's own signal stream waits on that event and launches a host
//!     function carrying the batch id. The compute stream never stalls on the
//!     callback, and compute streams never wait on each other's callbacks,
//!     because cuLaunchHostFunc runs in stream order on the signal stream.
//!  2. The host function runs on a CUDA driver thread. It only marks the id
//!     done and wakes the reaper; the driver must not be called from it.
//!  3. The worker pops done ids in order, runs the batch's handlers, and
//!     drops the batch, which releases the storage it retained.
//!  4. `wait_stream_idle(stream)` blocks until every batch committed on
//!     `stream` has run. `order_after_latest(source, waiter)` makes `waiter`
//!     wait on `source`'s latest event, once per commit.
//!
//! Retention lives in the pending map, never in a future: a stalled reaper
//! leaks, it never frees early. A batch whose callback could not be scheduled
//! is quarantined for the process lifetime and the commit reports the error,
//! without releasing in-flight resources. Callers check CUDA stream errors
//! before waiting for retirement; callbacks are not promised after a CUDA
//! context failure.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::batch::Batch;
use crate::driver::{self, CudaError, Stream};
use crate::event::Event;

pub struct Completion {
    state: Mutex<State>,
    /// Wakes the reaper task; the only thing the driver callback touches.
    wake: Condvar,
    /// Wakes `wait_stream_idle` callers after a batch has run.
    idle: Condvar,
}

struct State {
    /// Next batch id; ids increase in commit order across all streams.
    next_id: u64,
    /// Batches whose event has not run yet, by id.
    pending: BTreeMap<u64, Pending>,
    /// Ids the device has signaled, in signal order, awaiting the reaper.
    done: VecDeque<u64>,
    /// Batch the reaper is running right now, with its stream.
    running: Option<(u64, Stream)>,
    /// Batches that have run.
    completed: u64,
    /// Batches whose callback could not be scheduled; never run, never freed.
    quarantined: Vec<Pending>,
    /// Nonblocking stream per compute stream that carries only its host
    /// functions, created on first use and kept for the process.
    signals: HashMap<Stream, Stream>,
    /// Most recent commit per compute stream.
    latest: HashMap<Stream, Latest>,
    /// Latest id each (waiter, source) stream pair has already waited on.
    ordered: HashMap<(Stream, Stream), u64>,
}

struct Pending {
    device: i32,
    /// Compute stream the batch was committed on.
    stream: Stream,
    /// Retained values and handlers, run on the reaper.
    batch: Batch,
}

struct Latest {
    id: u64,
    /// Shared with waiters so the event is not pooled while they hold it.
    event: Arc<Event>,
}

impl Completion {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(State {
                next_id: 1,
                pending: BTreeMap::new(),
                done: VecDeque::new(),
                running: None,
                completed: 0,
                quarantined: Vec::new(),
                signals: HashMap::new(),
                latest: HashMap::new(),
                ordered: HashMap::new(),
            }),
            wake: Condvar::new(),
            idle: Condvar::new(),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("completion state is poisoned")
    }

    /// Record this commit on `stream` and hand `batch` to the reaper. Returns
    /// the batch id. An empty batch only records the stream's latest event.
    ///
    /// The batch enters the pending map before any driver call, so a failure
    /// anywhere in the enqueue quarantines it instead of dropping it while the
    /// device may still use what it retains.
    #[must_use = "the id identifies the batch in diagnostics"]
    pub fn commit(&self, device: i32, stream: Stream, batch: Batch) -> Result<u64, CudaError> {
        let has_batch = !batch.is_empty();
        let mut state = self.state();
        let id = state.next_id;
        state.next_id += 1;
        if has_batch {
            state.pending.insert(id, Pending { device, stream, batch });
        }
        drop(state);
        let mut callback_enqueued = false;
        let scheduled = self.record_latest(device, stream, id).and_then(|event| {
            if !has_batch {
                return Ok(());
            }
            let signal = self.signal_for(device, stream)?;
            driver::with_device(device, || {
                event.wait_on(signal)?;
                driver::launch_host_func(signal, signaled, id as usize)?;
                callback_enqueued = true;
                Ok(())
            })
        });
        if let Err(e) = scheduled {
            // A successful callback enqueue can retire the batch before the context is restored.
            if has_batch && !callback_enqueued {
                self.quarantine(id);
            }
            return Err(e);
        }
        Ok(id)
    }

    /// Record an event on `stream` and make it the stream's latest.
    fn record_latest(&self, device: i32, stream: Stream, id: u64) -> Result<Arc<Event>, CudaError> {
        let event = Event::take(device)?;
        event.record(stream)?;
        let event = Arc::new(event);
        self.state().latest.insert(stream, Latest { id, event: event.clone() });
        Ok(event)
    }

    fn signal_for(&self, device: i32, stream: Stream) -> Result<Stream, CudaError> {
        let mut state = self.state();
        if let Some(&signal) = state.signals.get(&stream) {
            return Ok(signal);
        }
        let signal = driver::with_device(device, driver::create_stream)?;
        state.signals.insert(stream, signal);
        Ok(signal)
    }

    /// The device may still be using what the batch retains and no callback
    /// will ever say otherwise, so keep it forever rather than free it.
    fn quarantine(&self, id: u64) {
        let mut state = self.state();
        let pending = state.pending.remove(&id).expect("failed batch must remain pending");
        state.quarantined.push(pending);
        drop(state);
        self.idle.notify_all();
    }

    /// Make `waiter` wait for everything committed on `source` so far. Each
    /// commit is waited on at most once per (waiter, source) pair.
    pub fn order_after_latest(&self, source: Stream, waiter: Stream) -> Result<(), CudaError> {
        if source == waiter {
            return Ok(());
        }
        let mut state = self.state();
        let Some(latest) = state.latest.get(&source) else {
            return Ok(());
        };
        let key = (waiter, source);
        if state.ordered.get(&key) == Some(&latest.id) {
            return Ok(());
        }
        let (id, event) = (latest.id, latest.event.clone());
        event.wait_on(waiter)?;
        state.ordered.insert(key, id);
        Ok(())
    }

    /// Block until no batch committed on `stream` is pending or running.
    pub fn wait_stream_idle(&self, stream: Stream) {
        let mut state = self.state();
        while state.has_work_on(stream) {
            state = self.idle.wait(state).expect("completion state is poisoned");
        }
    }

    /// Number of batches the reaper has run.
    pub fn completed_batches(&self) -> u64 {
        self.state().completed
    }

    /// The reaper: runs on its own runtime thread for the process lifetime.
    pub(crate) fn reap(&self) -> ! {
        loop {
            let pending = self.take_done();
            self.run(pending);
        }
    }

    fn take_done(&self) -> Pending {
        let mut state = self.state();
        while state.done.is_empty() {
            state = self.wake.wait(state).expect("completion state is poisoned");
        }
        let id = state.done.pop_front().expect("a signaled batch is available");
        let pending = state.pending.remove(&id).expect("signaled batch must remain pending");
        state.running = Some((id, pending.stream));
        pending
    }

    /// Run outside the lock: handlers release storage, which takes the
    /// allocator lock and may call back into `order_after_latest`.
    fn run(&self, pending: Pending) {
        let Pending { device, batch, .. } = pending;
        if let Err(error) = driver::with_device(device, || {
            batch.run();
            Ok(())
        }) {
            // A dead worker would leave every retirement waiter blocked.
            eprintln!("completion worker: {error}");
            std::process::abort();
        }
        let mut state = self.state();
        state.running = None;
        state.completed += 1;
        drop(state);
        self.idle.notify_all();
    }
}

impl State {
    fn has_work_on(&self, stream: Stream) -> bool {
        self.running.is_some_and(|(_, s)| s == stream)
            || self.pending.values().any(|p| p.stream == stream)
    }
}

/// Driver-thread callback: mark the batch done and wake the reaper. Nothing
/// else is allowed here because the driver forbids CUDA calls from host functions.
extern "C" fn signaled(data: *mut c_void) {
    let id = data as usize as u64;
    let completion = crate::runtime::completion();
    completion.state().done.push_back(id);
    completion.wake.notify_one();
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::Completion;

    #[test]
    fn invariant_poisoned_completion_state_is_rejected() {
        let completion = Completion::new();
        let interrupted = catch_unwind(AssertUnwindSafe(|| {
            let _guard = completion.state();
            panic!("interrupt a state update");
        }));
        assert!(interrupted.is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| drop(completion.state()))).is_err());
    }
}
