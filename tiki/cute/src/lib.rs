// Copyright © 2026 Dedalus Labs, Inc.

#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]

mod algebra;
mod error;
mod int;
mod layout;
mod offset;
mod param;
mod parse;
mod poly;
mod print;
mod shape;
mod stride;
mod swizzle;
mod tensor;
mod tiler;
mod truth;
mod tuple;
mod xor;

pub use error::{Condition, LayoutError, Verdict};
pub use int::Int;
pub use layout::Layout;
pub use offset::Offset;
pub use param::Param;
pub use print::{Cute, Description};
pub use shape::{Coord, Shape};
pub use stride::Stride;
pub use swizzle::{Swizzle, SwizzleParams};
pub use tensor::{Bounds, Tensor};
pub use tiler::Tiler;
pub use truth::Truth;
pub use tuple::{Profile, Tuple};
pub use xor::Xor;
