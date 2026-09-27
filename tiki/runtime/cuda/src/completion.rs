// Copyright © 2026 Dedalus Labs, Inc.

//! Completion runtime: each committed batch runs on a worker thread once the
//! device has finished the work enqueued before its commit.
//!
//! Protocol:
//!
//!  1. `commit(stream, batch)` makes the stream's own signal stream wait for
//!     everything enqueued on `stream` so far, then launches a host function
//!     on it carrying the batch id. The compute stream never stalls on the
//!     callback, and compute streams never wait on each other's callbacks,
//!     because cuLaunchHostFunc runs in stream order on the signal stream.
//!  2. The host function runs on a CUDA driver thread. It only marks the id
//!     done and wakes the worker; the driver must not be called from it.
//!  3. The worker pops done ids in order, runs the batch's handlers, and
//!     drops the batch, which releases the values it retained.
//!  4. `wait_stream_idle(stream)` blocks until every batch committed on
//!     `stream` has run.
//!
//! Retention lives in the pending map, never in a future: a stalled worker
//! leaks, it never frees early. A batch whose callback could not be scheduled
//! is quarantined for the process lifetime and the commit reports the error,
//! without releasing values the device may still use. Callbacks are not
//! promised after a CUDA context failure.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::batch::Batch;
use crate::driver::{CudaError, Device, Stream, StreamId};

pub struct Completion {
    state: Mutex<State>,
    /// Wakes the worker; the only thing the driver callback touches.
    wake: Condvar,
    /// Wakes `wait_stream_idle` callers after a batch has run.
    idle: Condvar,
}

struct State {
    /// Next batch id; ids increase in commit order across all streams.
    next_id: u64,
    /// Batches whose callback has not run yet, by id.
    pending: BTreeMap<u64, Pending>,
    /// Ids the device has signaled, in signal order, awaiting the worker.
    done: VecDeque<u64>,
    /// Batch the worker is running right now, with its stream.
    running: Option<(u64, StreamId)>,
    /// Batches that have run.
    completed: u64,
    /// Batches whose callback could not be scheduled; never run, never freed.
    quarantined: Vec<Pending>,
    /// Stream per compute stream that carries only its host functions,
    /// created on first use and kept for the process.
    signals: HashMap<(Device, StreamId), Arc<Stream>>,
}

struct Pending {
    /// Compute stream the batch was committed on.
    stream: StreamId,
    /// Retained values and handlers, run on the worker.
    batch: Batch,
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
            }),
            wake: Condvar::new(),
            idle: Condvar::new(),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("completion state is poisoned")
    }

    /// Hand `batch` to the worker once the device finishes the work enqueued
    /// on `stream` so far. Returns the batch id. An empty batch has nothing
    /// to run and enqueues nothing.
    ///
    /// The batch enters the pending map before any driver call, so a failure
    /// anywhere in the enqueue quarantines it instead of dropping it while the
    /// device may still use what it retains.
    #[must_use = "the id identifies the batch in diagnostics"]
    pub fn commit(&self, stream: &Stream, batch: Batch) -> Result<u64, CudaError> {
        let mut state = self.state();
        let id = state.next_id;
        state.next_id += 1;
        if batch.is_empty() {
            return Ok(id);
        }
        state.pending.insert(id, Pending { stream: stream.id(), batch });
        drop(state);
        let scheduled = (|| {
            let signal = self.signal_for(stream)?;
            signal.wait_for(stream)?;
            let payload = usize::try_from(id).expect("batch ids fit in usize");
            signal.launch_host_func(signaled, payload)
        })();
        if let Err(error) = scheduled {
            self.quarantine(id);
            return Err(error);
        }
        Ok(id)
    }

    fn signal_for(&self, stream: &Stream) -> Result<Arc<Stream>, CudaError> {
        let mut state = self.state();
        let key = (stream.device(), stream.id());
        if let Some(signal) = state.signals.get(&key) {
            return Ok(signal.clone());
        }
        let signal = Arc::new(Stream::new(stream.device())?);
        state.signals.insert(key, signal.clone());
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

    /// Block until no batch committed on `stream` is pending or running.
    pub fn wait_stream_idle(&self, stream: &Stream) {
        let mut state = self.state();
        while state.has_work_on(stream.id()) {
            state = self.idle.wait(state).expect("completion state is poisoned");
        }
    }

    /// Number of batches the worker has run.
    pub fn completed_batches(&self) -> u64 {
        self.state().completed
    }

    /// The worker: runs on its own runtime thread for the process lifetime.
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

    /// Run outside the lock: handlers release memory, which takes the
    /// allocator lock.
    fn run(&self, pending: Pending) {
        if catch_unwind(AssertUnwindSafe(|| pending.batch.run())).is_err() {
            // A dead worker would leave every retirement waiter blocked.
            eprintln!("completion worker: a batch handler panicked");
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
    fn has_work_on(&self, stream: StreamId) -> bool {
        self.running.is_some_and(|(_, running)| running == stream)
            || self.pending.values().any(|pending| pending.stream == stream)
    }
}

/// Driver-thread callback: mark the batch done and wake the worker. Nothing
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
