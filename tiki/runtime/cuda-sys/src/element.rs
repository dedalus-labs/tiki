// Copyright © 2026 Dedalus Labs, Inc.

//! Element types host memory may exchange with the device.

/// A plain numeric type with no padding and no invalid bit patterns, so any
/// bytes the device writes are a valid value. The trait is sealed: only the
/// types below implement it.
pub trait Element: Copy + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
}

macro_rules! elements {
    ($($type:ty),*) => {
        $(
            impl sealed::Sealed for $type {}
            impl Element for $type {}
        )*
    };
}

elements!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
