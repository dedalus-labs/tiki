// Copyright © 2026 Dedalus Labs, Inc.

//! Shapes and their coordinates (Cecka, Section 2.2).
//!
//! A shape is a hierarchical tuple of positive extents. Every shape has three kinds of
//! coordinate. An integer below the size is a 1-D coordinate. A tuple that follows the shape's
//! top-level modes is an n-D coordinate. A tuple congruent to the whole shape is the natural
//! coordinate, the one the strides act on. [`Shape::idx2crd`] refines any coordinate to the
//! natural one, splitting integers colexicographically: the leftmost mode varies fastest.

use crate::LayoutError;
use crate::int::Int;
use crate::offset::Offset;
use crate::stride::Stride;
use crate::tuple::{Profile, Tuple};
use std::fmt;

/// A coordinate: an integer, or a tuple weakly congruent to the shape it indexes.
pub type Coord = Tuple<Int>;

/// A hierarchical tuple of extents, each positive for every launch the parameter facts admit.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape(Tuple<Int>);

impl Shape {
    /// Wraps extents the algebra derived from checked shapes.
    pub(crate) fn from_derived(extents: Tuple<Int>) -> Self {
        Shape(extents)
    }

    pub fn as_tuple(&self) -> &Tuple<Int> {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    pub fn depth(&self) -> usize {
        self.0.depth()
    }

    pub fn profile(&self) -> Profile {
        self.0.profile()
    }

    pub fn extents(&self) -> Vec<&Int> {
        self.0.leaves()
    }

    pub fn modes(&self) -> impl Iterator<Item = Shape> + '_ {
        self.0.modes().iter().cloned().map(Shape)
    }

    /// Returns the number of coordinates, the product of the extents.
    pub fn size(&self) -> Int {
        self.extents().into_iter().fold(Int::Static(1), |product, extent| &product * extent)
    }

    /// Refines `coord` to the natural coordinate of this shape (Equation 4).
    ///
    /// An integer opposite a tuple splits by colexicographic division. The last leaf keeps the
    /// whole remaining quotient, so an index past the end lands past the end of the last mode,
    /// as in CuTe.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Coordinate`] when `coord` is not weakly congruent to the shape.
    pub fn idx2crd(&self, coord: &Coord) -> Result<Coord, LayoutError> {
        match (coord, &self.0) {
            (Tuple::Leaf(_), Tuple::Leaf(_)) => Ok(coord.clone()),
            (Tuple::Leaf(index), Tuple::Node(_)) => Ok(self.split(index)),
            (Tuple::Node(coords), Tuple::Node(modes)) if coords.len() == modes.len() => {
                let refined = coords.iter().zip(self.modes()).map(|(c, mode)| mode.idx2crd(c));
                Ok(Tuple::Node(refined.collect::<Result<_, _>>()?))
            }
            _ => Err(LayoutError::Coordinate { coord: coord.to_string(), shape: self.to_string() }),
        }
    }

    /// Splits a 1-D index over the extents, leftmost fastest. The last extent takes the rest.
    fn split(&self, index: &Int) -> Coord {
        let extents = self.extents();
        let mut rest = index.clone();
        let mut parts = Vec::with_capacity(extents.len());
        for (i, extent) in extents.iter().enumerate() {
            if i + 1 == extents.len() {
                parts.push(rest.clone());
            } else {
                parts.push(rest.rem_floor(extent));
                rest = rest.div_floor(extent);
            }
        }
        Tuple::from_leaves(&mut parts.into_iter(), &self.0)
    }

    /// Folds `coord` to its 1-D index (Equation 5), the inverse of [`Shape::idx2crd`].
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Coordinate`] when `coord` is not weakly congruent to the shape.
    pub fn crd2idx(&self, coord: &Coord) -> Result<Int, LayoutError> {
        // Each coordinate leaf stands for the whole subshape opposite it, so its weight is the
        // product of the sizes of the subshapes before it.
        let sizes =
            coord.zip(&self.0, &mut |_, sub| Shape(sub.clone()).size()).ok_or_else(|| {
                LayoutError::Coordinate { coord: coord.to_string(), shape: self.to_string() }
            })?;
        let weights = Shape(sizes).compact_stride(&Offset::from(1));
        let index = weights.inner_product(coord);
        Ok(index.as_int().expect("integer weights give an integer index").clone())
    }

    /// Returns the compact column-major strides of this shape, starting from `start`: each leaf's
    /// stride is `start` times the product of the extents before it.
    pub fn compact_stride(&self, start: &Offset) -> Stride {
        let mut running = start.clone();
        let strides: Vec<Offset> = self
            .extents()
            .into_iter()
            .map(|extent| {
                let stride = running.clone();
                running = running.scale(extent);
                stride
            })
            .collect();
        Stride::from_derived(Tuple::from_leaves(&mut strides.into_iter(), &self.0))
    }

    /// Returns whether every coordinate of this shape is a coordinate of `other`
    /// (Definition 2.7): each leaf's extent is the size of the subshape of `other` opposite it.
    pub fn is_compatible_with(&self, other: &Shape) -> bool {
        match (&self.0, &other.0) {
            (Tuple::Leaf(extent), _) => *extent == other.size(),
            (Tuple::Node(ours), Tuple::Node(theirs)) if ours.len() == theirs.len() => {
                self.modes().zip(other.modes()).all(|(a, b)| a.is_compatible_with(&b))
            }
            _ => false,
        }
    }
}

/// Checks that every extent is positive for every admitted launch.
impl TryFrom<Tuple<Int>> for Shape {
    type Error = LayoutError;

    fn try_from(extents: Tuple<Int>) -> Result<Self, LayoutError> {
        let not_positive = extents.leaves().into_iter().find(|e| !e.is_positive().is_proven());
        if let Some(extent) = not_positive {
            return Err(LayoutError::Extent { extent: extent.to_string() });
        }
        Ok(Shape(extents))
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
