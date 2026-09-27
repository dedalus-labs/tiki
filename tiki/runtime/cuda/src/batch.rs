// Copyright © 2026 Dedalus Labs, Inc.

//! What a commit keeps alive until the device has finished the work before it.

/// Values to retain and handlers to run once the device passes a commit. The
/// completion worker runs the handlers and drops the retained values on its
/// own thread, never on the driver's callback thread.
#[derive(Default)]
pub struct Batch {
    handlers: Vec<Box<dyn FnOnce() + Send>>,
}

impl Batch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep `value` alive until the commit completes.
    pub fn retain<T: Send + 'static>(&mut self, value: T) {
        self.handlers.push(Box::new(move || drop(value)));
    }

    /// Run `handler` once the commit completes.
    pub fn on_complete(&mut self, handler: impl FnOnce() + Send + 'static) {
        self.handlers.push(Box::new(handler));
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    pub(crate) fn run(self) {
        for handler in self.handlers {
            handler();
        }
    }
}
