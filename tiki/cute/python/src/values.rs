// Copyright © 2026 Dedalus Labs, Inc.

//! Stride values that are not integers: [`F2`], an XOR value, and [`ArithTuple`], a coordinate
//! offset. Both print as PyCuTe prints them, because the `tiki.layout` API prints layouts that way.

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyIterator, PyTuple};
use tiki_cute::{Int, Offset, Tuple, Xor};

use crate::convert::{integer, integers, offset_to_python};
use crate::{LayoutError, refused};

/// An XOR stride value, PyCuTe's `F2`: `+` is XOR and a product is carry-less.
///
/// Adding a nonzero integer is a `TypeError`, since carrying addition does not mix with XOR. An
/// integer multiplies an `F2` through its bits, so `3 * F2(3)` is `F2(5)`.
#[pyclass(frozen, module = "tiki.layout._cute", name = "F2")]
#[derive(Clone, Copy)]
pub struct F2(Xor);

impl F2 {
    /// Returns the XOR value.
    pub fn xor(&self) -> Xor {
        self.0
    }
}

impl From<Xor> for F2 {
    fn from(value: Xor) -> Self {
        F2(value)
    }
}

/// The right operand of an `F2` or `ArithTuple` operation.
enum Operand {
    /// An `F2`.
    Xor(Xor),
    /// A Python integer that fits an `i64`.
    Integer(i64),
    /// A value the operation does not know, so Python tries the reflected operation.
    Other,
}

impl Operand {
    fn of(value: &Bound<'_, PyAny>) -> Operand {
        if let Ok(xor) = value.cast::<F2>() {
            return Operand::Xor(xor.get().0);
        }
        if value.is_instance_of::<PyBool>() {
            return Operand::Other;
        }
        value.extract::<i64>().map_or(Operand::Other, Operand::Integer)
    }
}

/// Returns the carry-less product, or refuses one that needs more than 63 bits.
fn product(a: Xor, b: Xor) -> PyResult<Xor> {
    let width = |value: Xor| i64::BITS - value.value().leading_zeros();
    if a.value() != 0 && b.value() != 0 && width(a) + width(b) > i64::BITS {
        return Err(LayoutError::new_err(format!(
            "F{} * F{} overflows 63 bits",
            a.value(),
            b.value()
        )));
    }
    Ok(a * b)
}

// PyO3 methods take `&self`, however small the value.
#[allow(clippy::trivially_copy_pass_by_ref)]
#[pymethods]
impl F2 {
    /// Returns the XOR value with the bits of a nonnegative integer or of another `F2`.
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(xor) = value.cast::<F2>() {
            return Ok(*xor.get());
        }
        let bits = integer(value, "an F2 value")?;
        Xor::try_from(bits).map(F2).map_err(|error| refused(&error))
    }

    /// The bits as a nonnegative integer.
    #[getter]
    fn value(&self) -> i64 {
        self.0.value()
    }

    fn __int__(&self) -> i64 {
        self.0.value()
    }

    fn __repr__(&self) -> String {
        format!("F{}", self.0.value())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Hashes as the integer with the same bits, which compares equal to it.
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.0.value().into_pyobject(py)?.hash()
    }

    /// Orders by value, so a sort by stride visits XOR strides by leading bit, as in PyCuTe.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> Py<PyAny> {
        let py = other.py();
        let value = match Operand::of(other) {
            Operand::Xor(xor) => xor.value(),
            Operand::Integer(value) => value,
            Operand::Other => return py.NotImplemented(),
        };
        PyBool::new(py, op.matches(self.0.value().cmp(&value))).to_owned().into_any().unbind()
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        match Operand::of(other) {
            Operand::Xor(xor) => Ok(Bound::new(py, F2(self.0 + xor))?.into_any().unbind()),
            Operand::Integer(0) => Ok(Bound::new(py, *self)?.into_any().unbind()),
            Operand::Integer(value) => {
                Err(PyTypeError::new_err(format!("F2 adds only F2 values and 0, got {value}")))
            }
            Operand::Other => Ok(py.NotImplemented()),
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__add__(other)
    }

    /// XOR is its own inverse, so subtraction is addition.
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__add__(other)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__add__(other)
    }

    fn __neg__(&self) -> F2 {
        *self
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let factor = match Operand::of(other) {
            Operand::Xor(xor) => xor,
            Operand::Integer(value) => Xor::try_from(value).map_err(|error| refused(&error))?,
            Operand::Other => return Ok(py.NotImplemented()),
        };
        Ok(Bound::new(py, F2(product(self.0, factor)?))?.into_any().unbind())
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__mul__(other)
    }
}

/// A coordinate offset, PyCuTe's `ArithTuple`: an element of Z^n written as a sum of scaled
/// basis elements, such as `2@0` for `2 * E(0)`.
///
/// It is never an integer or an XOR value, which convert to `int` and [`F2`]. Addition is
/// componentwise, the missing trailing components are 0, and an integer multiplies every
/// component. It compares equal to a tuple of the same components.
#[pyclass(frozen, module = "tiki.layout._cute", name = "ArithTuple")]
pub struct ArithTuple(Offset);

impl ArithTuple {
    /// Wraps a coordinate offset. The caller converts integers and XOR values instead.
    pub(crate) fn new(offset: Offset) -> Self {
        debug_assert!(offset.as_int().is_none() && offset.as_xor().is_none());
        ArithTuple(offset)
    }

    /// Returns the offset.
    pub fn offset(&self) -> &Offset {
        &self.0
    }

    fn components(&self) -> &[Tuple<Int>] {
        self.0.as_tuple().expect("an ArithTuple is not an XOR value").modes()
    }

    /// Reads the other operand of an arithmetic or comparison operation as integer components.
    /// An `F2` or another type is `None`, so Python tries the reflected operation.
    fn operand(other: &Bound<'_, PyAny>) -> PyResult<Option<Tuple<Int>>> {
        if let Ok(coordinates) = other.cast::<ArithTuple>() {
            return Ok(coordinates.get().0.as_tuple().cloned());
        }
        if other.cast::<F2>().is_ok() || other.is_instance_of::<PyBool>() {
            return Ok(None);
        }
        if other.extract::<i64>().is_ok() || other.cast::<PyTuple>().is_ok() {
            return integers(other, "a coordinate").map(Some);
        }
        Ok(None)
    }
}

/// Adds componentwise, reading missing trailing components as 0.
fn add(a: &Tuple<Int>, b: &Tuple<Int>) -> PyResult<Tuple<Int>> {
    let zero = Tuple::Leaf(Int::Static(0));
    match (a, b) {
        (Tuple::Leaf(x), Tuple::Leaf(y)) => {
            let (x, y) = (static_value(x)?, static_value(y)?);
            let sum = x.checked_add(y).ok_or_else(|| overflow(x, "+", y))?;
            Ok(Tuple::Leaf(Int::Static(sum)))
        }
        (Tuple::Leaf(_), Tuple::Node(_)) if *a == zero => Ok(b.clone()),
        (Tuple::Node(_), Tuple::Leaf(_)) if *b == zero => Ok(a.clone()),
        (Tuple::Node(xs), Tuple::Node(ys)) => {
            let rank = xs.len().max(ys.len());
            let component = |modes: &[Tuple<Int>], i: usize| modes.get(i).unwrap_or(&zero).clone();
            let sums = (0..rank).map(|i| add(&component(xs, i), &component(ys, i)));
            sums.collect::<PyResult<Vec<_>>>().map(Tuple::Node)
        }
        _ => Err(PyTypeError::new_err(format!(
            "{} and {} lie in different codomains",
            pycute(a),
            pycute(b)
        ))),
    }
}

/// Multiplies every component by `factor`.
fn scale(a: &Tuple<Int>, factor: i64) -> PyResult<Tuple<Int>> {
    match a {
        Tuple::Leaf(x) => {
            let x = static_value(x)?;
            let product = x.checked_mul(factor).ok_or_else(|| overflow(x, "*", factor))?;
            Ok(Tuple::Leaf(Int::Static(product)))
        }
        Tuple::Node(modes) => {
            modes.iter().map(|mode| scale(mode, factor)).collect::<PyResult<_>>().map(Tuple::Node)
        }
    }
}

fn static_value(value: &Int) -> PyResult<i64> {
    value.as_static().ok_or_else(|| LayoutError::new_err(format!("{value} is not static")))
}

fn overflow(a: i64, operation: &str, b: i64) -> PyErr {
    LayoutError::new_err(format!("{a} {operation} {b} overflows a signed 64-bit integer"))
}

/// Prints PyCuTe's form: a single term `c * E(p0, .., pn)` as `c@pn@..@p0`, any other offset
/// as its components.
fn pycute(tuple: &Tuple<Int>) -> String {
    let terms = Offset::from_tuple(tuple.clone()).terms();
    if let [(value, path)] = terms.as_slice() {
        return path.iter().rev().fold(value.to_string(), |text, mode| format!("{text}@{mode}"));
    }
    match tuple {
        Tuple::Leaf(value) => value.to_string(),
        Tuple::Node(modes) => {
            let components: Vec<String> = modes.iter().map(pycute).collect();
            format!("({})", components.join(","))
        }
    }
}

/// Returns the canonical offset of `tuple` as an `int` or an `ArithTuple`.
fn to_python(py: Python<'_>, tuple: Tuple<Int>) -> PyResult<Py<PyAny>> {
    Ok(offset_to_python(py, &Offset::from_tuple(tuple))?.unbind())
}

#[pymethods]
impl ArithTuple {
    fn __repr__(&self) -> String {
        pycute(self.0.as_tuple().expect("an ArithTuple is not an XOR value"))
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __len__(&self) -> usize {
        self.components().len()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let components = self.components();
        let position =
            if index < 0 { index.checked_add_unsigned(components.len()) } else { Some(index) };
        let component = position
            .and_then(|position| usize::try_from(position).ok())
            .and_then(|position| components.get(position))
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("ArithTuple index out of range")
            })?;
        offset_to_python(py, &Offset::from_tuple(component.clone()))
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let components = self
            .components()
            .iter()
            .map(|component| offset_to_python(py, &Offset::from_tuple(component.clone())));
        PyTuple::new(py, components.collect::<PyResult<Vec<_>>>()?)?.try_iter()
    }

    /// Compares componentwise with an `ArithTuple`, an integer or a tuple. Order is
    /// colexicographic, last component first.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(other) = ArithTuple::operand(other)? else { return Ok(py.NotImplemented()) };
        let other = Offset::from_tuple(other);
        let result = match op {
            CompareOp::Eq => self.0 == other,
            CompareOp::Ne => self.0 != other,
            _ => {
                let order = self.0.compare(&other).ok_or_else(|| {
                    PyTypeError::new_err(format!("{} and {other} have no order", self.__repr__()))
                })?;
                op.matches(order)
            }
        };
        Ok(PyBool::new(py, result).to_owned().into_any().unbind())
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(other) = ArithTuple::operand(other)? else { return Ok(py.NotImplemented()) };
        to_python(py, add(self.0.as_tuple().expect("not XOR"), &other)?)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(other) = ArithTuple::operand(other)? else { return Ok(py.NotImplemented()) };
        to_python(py, add(self.0.as_tuple().expect("not XOR"), &scale(&other, -1)?)?)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(other) = ArithTuple::operand(other)? else { return Ok(py.NotImplemented()) };
        to_python(py, add(&other, &scale(self.0.as_tuple().expect("not XOR"), -1)?)?)
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, scale(self.0.as_tuple().expect("not XOR"), -1)?)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Operand::Integer(factor) = Operand::of(other) else { return Ok(py.NotImplemented()) };
        to_python(py, scale(self.0.as_tuple().expect("not XOR"), factor)?)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__mul__(other)
    }
}

#[cfg(test)]
mod tests {
    use super::pycute;
    use tiki_cute::{Int, Offset, Tuple};

    fn tuple(offset: &Offset) -> Tuple<Int> {
        offset.as_tuple().cloned().expect("an arithmetic offset")
    }

    #[test]
    fn prints_as_pycute() {
        assert_eq!(pycute(&tuple(&Offset::basis(&[0]))), "1@0");
        assert_eq!(pycute(&tuple(&Offset::basis(&[1, 0]))), "1@0@1");
        assert_eq!(
            pycute(&tuple(&Offset::basis(&[0]).add(&Offset::basis(&[1]).scale(&Int::Static(3))))),
            "(1,3)"
        );
        assert_eq!(pycute(&tuple(&Offset::zero())), "0");
    }
}
