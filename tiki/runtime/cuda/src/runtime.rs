// Copyright © 2026 Dedalus Labs, Inc.

//! Composition root: constructs the allocator and the completion runtime once
//! and starts the reaper thread.

use std::sync::OnceLock;

use crate::allocator::Allocator;
use crate::completion::Completion;
use crate::driver::CudaError;

static ALLOCATOR: OnceLock<Allocator> = OnceLock::new();
static COMPLETION: OnceLock<Completion> = OnceLock::new();

/// Construct the process-wide runtime. Later calls are no-ops.
pub fn init() -> Result<(), CudaError> {
    if ALLOCATOR.get().is_none() {
        let _ = ALLOCATOR.set(Allocator::new()?);
    }
    if COMPLETION.get().is_none() && COMPLETION.set(Completion::new()).is_ok() {
        start_reaper();
    }
    Ok(())
}

pub fn allocator() -> &'static Allocator {
    ALLOCATOR.get().expect("tiki-cuda-runtime: init() was not called")
}

pub fn completion() -> &'static Completion {
    COMPLETION.get().expect("tiki-cuda-runtime: init() was not called")
}

/// Completion callbacks wake this process-lifetime worker; it releases batches
/// outside the CUDA callback thread.
fn start_reaper() {
    let spawned = std::thread::Builder::new()
        .name("tiki-cuda-reaper".into())
        .spawn(|| -> () { completion().reap() });
    spawned.expect("tiki-cuda-runtime: cannot spawn the reaper thread");
}
