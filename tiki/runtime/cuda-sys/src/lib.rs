// Copyright © 2026 Dedalus Labs, Inc.

//! Tiki's only call site into the CUDA driver (`libcuda`), and the only Tiki
//! crate that contains `unsafe`. The declarations in `sys` are generated from
//! `cuda.h` by `bindings.sh`.
//!
//! Every value this crate hands out owns what it names: a [`Stream`], an
//! [`Event`], or [`DeviceMemory`]. No address or driver handle crosses the
//! crate as an integer, so safe code cannot name memory it does not own. Host
//! memory reaches the driver only as a borrowed slice, for the length of a
//! copy that finishes before the call returns.
//!
//! Device memory records the events of its last write and of its reads since.
//! Every operation on it waits on them first, and dropping it frees it after
//! its last use, in stream order.
//!
//! Each driver call runs with its device's primary context pushed, and the
//! caller's context is restored afterwards, also when the call fails or
//! panics.

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]
mod sys;

mod context;
mod device;
mod element;
mod error;
mod event;
mod memory;
mod stream;

pub use device::{Device, device_count};
pub use element::Element;
pub use error::{CopyError, CudaError};
pub use event::Event;
pub use memory::DeviceMemory;
pub use stream::{Stream, StreamId};

// Device addresses and handles are stored as 64-bit integers.
const _: () = assert!(usize::BITS == 64);
