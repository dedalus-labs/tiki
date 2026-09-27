// Copyright © 2026 Dedalus Labs, Inc.

//! [`Swizzle`], tiki-cute's validated XOR index transform.

use pyo3::prelude::*;
use tiki_cute::SwizzleParams;

use crate::refused;

/// An immutable swizzle, validated by tiki-cute when it is built.
///
/// `tiki.layout.Swizzle` subclasses it to take keyword parameters and to print its constructor.
#[pyclass(subclass, frozen, eq, hash, module = "tiki.layout._cute", name = "Swizzle")]
#[derive(PartialEq, Eq, Hash)]
pub struct Swizzle(tiki_cute::Swizzle);

#[pymethods]
impl Swizzle {
    /// Checks the fields: they must not overlap, and both must fit bits 0 through 62.
    #[new]
    fn new(bits: i64, base: i64, shift: i64) -> PyResult<Self> {
        let params = SwizzleParams { bits, base, shift };
        tiki_cute::Swizzle::try_from(params).map(Swizzle).map_err(|error| refused(&error))
    }

    /// Width of each bit field.
    #[getter]
    fn bits(&self) -> u32 {
        self.0.bits()
    }

    /// Number of untouched low bits below both fields.
    #[getter]
    fn base(&self) -> u32 {
        self.0.base()
    }

    /// Signed distance from the destination field to the source field.
    #[getter]
    fn shift(&self) -> i32 {
        self.0.shift()
    }

    /// Transforms a nonnegative element offset.
    fn __call__(&self, index: i64) -> PyResult<i64> {
        self.0.apply(index).map_err(|error| refused(&error))
    }
}
