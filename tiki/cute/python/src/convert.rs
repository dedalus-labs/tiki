// Copyright © 2026 Dedalus Labs, Inc.

//! Conversions between Python values and tiki-cute's typed values.
//!
//! Each Python form maps to one Rust type: nested tuples or lists of integers to [`Tuple<Int>`],
//! stride values to [`Offset`], `None` or `slice(None)` to a free coordinate, and a tiler of
//! layouts, extents and `None` holes to [`Tiler`]. A value outside these forms is refused here,
//! before the algebra sees it.

use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PySlice, PyTuple};
use tiki_cute::{Int, Offset, Shape, Tiler, Tuple};

use crate::layout::Layout;
use crate::values::{ArithTuple, F2};
use crate::{LayoutError, refused};

/// Reads a Python integer that fits an `i64`. A `bool` is refused, since it is not an extent,
/// stride or coordinate.
pub(crate) fn integer(value: &Bound<'_, PyAny>, what: &str) -> PyResult<i64> {
    if value.is_instance_of::<PyBool>() {
        return Err(LayoutError::new_err(format!("{what} must be an integer, got {value}")));
    }
    value.extract::<i64>().map_err(|error| {
        let reason = if error.is_instance_of::<PyOverflowError>(value.py()) {
            "fit a signed 64-bit integer"
        } else {
            "be an integer"
        };
        LayoutError::new_err(format!("{what} must {reason}, got {value}"))
    })
}

/// Returns the items of a tuple or list, the two node types of a hierarchical tuple, or `None`
/// for a leaf.
fn modes<'py>(value: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return Some(tuple.iter().collect());
    }
    value.cast::<PyList>().ok().map(|list| list.iter().collect())
}

/// Reads a hierarchical tuple, reading each leaf with `leaf`.
fn hierarchical<T>(
    value: &Bound<'_, PyAny>,
    leaf: &mut impl FnMut(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Tuple<T>> {
    match modes(value) {
        None => leaf(value).map(Tuple::Leaf),
        Some(items) => {
            let modes = items.iter().map(|item| hierarchical(item, leaf));
            modes.collect::<PyResult<Vec<_>>>().map(Tuple::Node)
        }
    }
}

/// Reads a hierarchical tuple of integers, such as an extent tuple or a coordinate.
pub(crate) fn integers(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Tuple<Int>> {
    hierarchical(value, &mut |leaf| integer(leaf, what).map(Int::Static))
}

/// Reads a shape: a hierarchical tuple of positive integers.
pub(crate) fn shape(value: &Bound<'_, PyAny>) -> PyResult<Shape> {
    Shape::try_from(integers(value, "an extent")?).map_err(|error| refused(&error))
}

/// Reads one stride value: an integer, an [`F2`] or an [`ArithTuple`].
pub(crate) fn offset(value: &Bound<'_, PyAny>) -> PyResult<Offset> {
    if let Ok(xor) = value.cast::<F2>() {
        return Ok(Offset::from(xor.get().xor()));
    }
    if let Ok(coordinates) = value.cast::<ArithTuple>() {
        return Ok(coordinates.get().offset().clone());
    }
    integer(value, "a stride").map(Offset::from)
}

/// Reads a hierarchical tuple of stride values.
pub(crate) fn offsets(value: &Bound<'_, PyAny>) -> PyResult<Tuple<Offset>> {
    hierarchical(value, &mut offset)
}

/// Reads a slicing coordinate, where `None` or `slice(None)` leaves a mode free.
pub(crate) fn slicing(value: &Bound<'_, PyAny>) -> PyResult<Tuple<Option<Int>>> {
    hierarchical(value, &mut |leaf| {
        if leaf.is_none() || is_whole_slice(leaf)? {
            return Ok(None);
        }
        integer(leaf, "a coordinate").map(|index| Some(Int::Static(index)))
    })
}

/// Returns whether `value` is `slice(None)`, the only slice that names a whole mode.
fn is_whole_slice(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Ok(slice) = value.cast::<PySlice>() else { return Ok(false) };
    for bound in ["start", "stop", "step"] {
        if !slice.getattr(bound)?.is_none() {
            return Err(LayoutError::new_err(format!(
                "a coordinate slices whole modes only, got {value}"
            )));
        }
    }
    Ok(true)
}

/// Reads a tiler: a [`Layout`], an extent `n` for `n:1`, or a tuple of tilers with `None` for
/// each mode it leaves unchanged.
pub(crate) fn tiler(value: &Bound<'_, PyAny>) -> PyResult<Tiler> {
    if let Ok(layout) = value.cast::<Layout>() {
        return Ok(Tiler::Layout(layout.get().value().clone()));
    }
    if let Some(items) = modes(value) {
        let modes =
            items.iter().map(|item| if item.is_none() { Ok(None) } else { tiler(item).map(Some) });
        return modes.collect::<PyResult<Vec<_>>>().map(Tiler::Modes);
    }
    Tiler::try_from(integer(value, "a tiler extent")?).map_err(|error| refused(&error))
}

/// Returns a static integer as a Python `int`. Values built from Python are always static.
fn static_int<'py>(py: Python<'py>, value: &Int) -> PyResult<Bound<'py, PyAny>> {
    let value = value
        .as_static()
        .ok_or_else(|| LayoutError::new_err(format!("{value} depends on a launch parameter")))?;
    Ok(value.into_pyobject(py)?.into_any())
}

/// Returns a hierarchical tuple as nested Python tuples, converting each leaf with `leaf`.
fn to_python<'py, T>(
    py: Python<'py>,
    value: &Tuple<T>,
    leaf: &impl Fn(Python<'py>, &T) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Tuple::Leaf(value) => leaf(py, value),
        Tuple::Node(modes) => {
            let modes = modes.iter().map(|mode| to_python(py, mode, leaf));
            Ok(PyTuple::new(py, modes.collect::<PyResult<Vec<_>>>()?)?.into_any())
        }
    }
}

/// Returns a hierarchical tuple of integers as nested Python tuples.
pub(crate) fn integers_to_python<'py>(
    py: Python<'py>,
    value: &Tuple<Int>,
) -> PyResult<Bound<'py, PyAny>> {
    to_python(py, value, &static_int)
}

/// Returns a stride value as an `int`, an [`F2`] or an [`ArithTuple`].
pub(crate) fn offset_to_python<'py>(
    py: Python<'py>,
    value: &Offset,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(xor) = value.as_xor() {
        return Ok(Bound::new(py, F2::from(xor))?.into_any());
    }
    match value.as_int() {
        Some(integer) => static_int(py, integer),
        None => Ok(Bound::new(py, ArithTuple::new(value.clone()))?.into_any()),
    }
}

/// Returns a hierarchical tuple of stride values as nested Python tuples.
pub(crate) fn offsets_to_python<'py>(
    py: Python<'py>,
    value: &Tuple<Offset>,
) -> PyResult<Bound<'py, PyAny>> {
    to_python(py, value, &offset_to_python)
}
