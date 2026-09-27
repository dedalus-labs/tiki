// Copyright © 2026 Dedalus Labs, Inc.

//! [`Layout`], a handle to a checked tiki-cute layout, with the algebra the `tiki.layout` Python
//! layer calls, and the shape functions it calls on bare tuples.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use tiki_cute::Stride;

use crate::convert::{
    self, integers, integers_to_python, offset, offset_to_python, offsets, offsets_to_python,
    slicing, tiler,
};
use crate::values::{ArithTuple, F2};
use crate::{LayoutError, refused};

/// A layout: a shape and a congruent stride, checked by tiki-cute when it is built.
///
/// Python builds one from nested tuples and reads its shape and stride back as nested tuples.
/// Every operation returns a new handle or raises `LayoutError`, and none mutates one.
#[pyclass(frozen, eq, hash, module = "tiki.layout._cute", name = "Layout")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Layout(tiki_cute::Layout);

impl Layout {
    /// Returns the tiki-cute layout.
    pub fn value(&self) -> &tiki_cute::Layout {
        &self.0
    }
}

impl From<tiki_cute::Layout> for Layout {
    fn from(layout: tiki_cute::Layout) -> Self {
        Layout(layout)
    }
}

/// Wraps the result of an algebra operation, raising a refusal as `LayoutError`.
fn checked(result: Result<tiki_cute::Layout, tiki_cute::LayoutError>) -> PyResult<Layout> {
    result.map(Layout).map_err(|error| refused(&error))
}

#[pymethods]
impl Layout {
    /// Builds `shape:stride` from a tuple of positive extents and a weakly congruent stride. A
    /// stride leaf opposite a subshape starts that subshape's compact strides.
    #[new]
    fn new(shape: &Bound<'_, PyAny>, stride: &Bound<'_, PyAny>) -> PyResult<Self> {
        let stride = Stride::try_from(offsets(stride)?).map_err(|error| refused(&error))?;
        checked(tiki_cute::Layout::new(convert::shape(shape)?, stride))
    }

    /// Concatenates layouts as the top-level modes of one layout.
    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)] // PyO3 extracts a sequence argument by value.
    fn from_modes(modes: Vec<PyRef<'_, Layout>>) -> PyResult<Self> {
        checked(tiki_cute::Layout::try_from_modes(modes.iter().map(|mode| mode.0.clone())))
    }

    /// Returns the layout that composes as `tiler` does, PyCuTe's `tiler_to_layout`.
    #[staticmethod]
    fn from_tiler(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        checked(tiler(value)?.to_layout())
    }

    /// The extents, as nested tuples of integers.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        integers_to_python(py, self.0.shape().as_tuple())
    }

    /// The strides, as nested tuples of integers, `F2` values or `ArithTuple` values.
    #[getter]
    fn stride<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        offsets_to_python(py, self.0.stride().as_tuple())
    }

    /// Number of top-level modes. A leaf layout has rank 1.
    #[getter]
    fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Levels of nesting of the shape. A leaf layout has depth 0.
    #[getter]
    fn depth(&self) -> usize {
        self.0.depth()
    }

    /// One past the offset of the last coordinate along each codomain axis.
    fn coshape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        integers_to_python(py, &self.0.coshape())
    }

    /// Returns the sublayout at `path`, one mode index per level.
    #[allow(clippy::needless_pass_by_value)] // PyO3 extracts a sequence argument by value.
    fn get(&self, path: Vec<usize>) -> PyResult<Self> {
        self.0
            .get(&path)
            .map(Layout)
            .ok_or_else(|| LayoutError::new_err(format!("{} has no mode {path:?}", self.0.cute())))
    }

    /// Returns the offset of a coordinate: an index, a tuple of indices, or a natural coordinate.
    /// An `F2` or `ArithTuple`, another layout's offset, reads as a coordinate of this layout.
    fn __call__<'py>(&self, coordinate: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let result = if coordinate.cast::<F2>().is_ok() || coordinate.cast::<ArithTuple>().is_ok() {
            self.0.call_offset(&offset(coordinate)?)
        } else {
            self.0.call(&integers(coordinate, "a coordinate")?)
        };
        offset_to_python(coordinate.py(), &result.map_err(|error| refused(&error))?)
    }

    /// Splits a coordinate into the offset of its fixed leaves and the layout of its free ones,
    /// where `None` leaves a mode free.
    fn slice<'py>(&self, coordinate: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Layout)> {
        let (fixed, free) = self.0.slice(&slicing(coordinate)?).map_err(|error| refused(&error))?;
        Ok((offset_to_python(coordinate.py(), &fixed)?, Layout(free)))
    }

    /// Returns the flat layout with the same function on every index of its domain.
    fn coalesce(&self) -> Self {
        Layout(self.0.coalesce())
    }

    /// Returns `self ∘ tiler`.
    fn compose(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        checked(self.0.compose(&tiler(value)?))
    }

    fn logical_divide(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        checked(self.0.logical_divide(&tiler(value)?))
    }

    fn zipped_divide(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        checked(self.0.zipped_divide(&tiler(value)?))
    }

    fn logical_product(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        checked(self.0.logical_product(&tiler(value)?))
    }

    fn blocked_product(&self, tile: &Bound<'_, Layout>) -> PyResult<Self> {
        checked(self.0.blocked_product(&tile.get().0))
    }

    fn raked_product(&self, tile: &Bound<'_, Layout>) -> PyResult<Self> {
        checked(self.0.raked_product(&tile.get().0))
    }

    /// Returns the complement, grown to cover `cotarget` when it is given.
    #[pyo3(signature = (cotarget=None))]
    fn complement(&self, cotarget: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        match cotarget {
            None => checked(self.0.complement()),
            Some(cotarget) => checked(self.0.complement_in(&convert::shape(cotarget)?)),
        }
    }

    fn right_inverse(&self) -> Self {
        Layout(self.0.right_inverse())
    }

    fn left_inverse(&self) -> PyResult<Self> {
        checked(self.0.left_inverse())
    }

    fn nullspace(&self) -> Self {
        Layout(self.0.nullspace())
    }

    /// Returns the layout over elements `scale` times as wide, one integer factor per codomain
    /// axis.
    fn recast(&self, scale: &Bound<'_, PyAny>) -> PyResult<Self> {
        checked(self.0.recast(&integers(scale, "a recast factor")?))
    }

    fn __repr__(&self) -> String {
        format!("_cute.Layout({})", self.0.cute())
    }
}

/// Returns the unit basis element `E(path)`: the integer 1 for the empty path, and an
/// [`ArithTuple`] otherwise.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts a sequence argument by value.
pub fn basis(py: Python<'_>, path: Vec<usize>) -> PyResult<Bound<'_, PyAny>> {
    offset_to_python(py, &tiki_cute::Offset::basis(&path))
}

/// Returns the natural coordinate of `coordinate` in `shape`. An `F2` index splits over the
/// shape carry-lessly.
#[pyfunction]
pub fn idx2crd<'py>(
    coordinate: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = coordinate.py();
    let shape = convert::shape(shape)?;
    if let Ok(xor) = coordinate.cast::<F2>() {
        let natural = shape.idx2crd_xor(xor.get().xor()).map_err(|error| refused(&error))?;
        return xors_to_python(py, &natural);
    }
    let natural =
        shape.idx2crd(&integers(coordinate, "a coordinate")?).map_err(|error| refused(&error))?;
    integers_to_python(py, &natural)
}

fn xors_to_python<'py>(
    py: Python<'py>,
    value: &tiki_cute::Tuple<tiki_cute::Xor>,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        tiki_cute::Tuple::Leaf(xor) => Ok(Bound::new(py, F2::from(*xor))?.into_any()),
        tiki_cute::Tuple::Node(modes) => {
            let modes = modes.iter().map(|mode| xors_to_python(py, mode));
            Ok(PyTuple::new(py, modes.collect::<PyResult<Vec<_>>>()?)?.into_any())
        }
    }
}

/// Returns the index of `coordinate` in `shape`, the inverse of [`idx2crd`].
#[pyfunction]
pub fn crd2idx<'py>(
    coordinate: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let index = convert::shape(shape)?
        .crd2idx(&integers(coordinate, "a coordinate")?)
        .map_err(|error| refused(&error))?;
    offset_to_python(coordinate.py(), &tiki_cute::Offset::from(index))
}

/// Returns whether every coordinate of shape `a` is a coordinate of shape `b`.
#[pyfunction]
pub fn compatible(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(convert::shape(a)?.is_compatible_with(&convert::shape(b)?))
}
