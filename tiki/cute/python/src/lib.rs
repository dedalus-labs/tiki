// Copyright © 2026 Dedalus Labs, Inc.

//! The `tiki.layout._cute` extension: tiki-cute's layout algebra behind a typed Python interface.
//!
//! Python passes typed values only: nested tuples of integers, [`F2`] and [`ArithTuple`] stride
//! values, tilers of layouts, and [`Layout`] handles. Nothing is formatted in Python and parsed
//! here. Every refusal of the algebra becomes `tiki.layout._cute.LayoutError`, a `ValueError`.

#![forbid(unsafe_code)]

mod convert;
mod layout;
mod swizzle;
mod values;

pub use layout::Layout;
pub use swizzle::Swizzle;
pub use values::{ArithTuple, F2};

use pyo3::prelude::*;

pyo3::create_exception!(
    tiki.layout._cute,
    LayoutError,
    pyo3::exceptions::PyValueError,
    "A layout, tensor or swizzle operation refused its operands."
);

/// Raises a tiki-cute refusal as [`LayoutError`] with the refusal's message.
pub(crate) fn refused(error: &tiki_cute::LayoutError) -> PyErr {
    LayoutError::new_err(error.to_string())
}

#[pymodule(gil_used = false)]
fn _cute(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("LayoutError", module.py().get_type::<LayoutError>())?;
    module.add_class::<Layout>()?;
    module.add_class::<F2>()?;
    module.add_class::<ArithTuple>()?;
    module.add_class::<Swizzle>()?;
    module.add_function(wrap_pyfunction!(layout::basis, module)?)?;
    module.add_function(wrap_pyfunction!(layout::idx2crd, module)?)?;
    module.add_function(wrap_pyfunction!(layout::crd2idx, module)?)?;
    module.add_function(wrap_pyfunction!(layout::compatible, module)?)?;
    Ok(())
}
