// Copyright © 2026 Dedalus Labs, Inc.

//! Layouts: functions from the coordinates of a shape to codomain offsets, `L = D ∘ S` (Cecka,
//! Section 2.3). `S` refines a coordinate to the natural one and `D` takes its inner product
//! with the strides.

use crate::LayoutError;
use crate::int::Int;
use crate::offset::Offset;
use crate::param::Param;
use crate::shape::{Coord, Shape};
use crate::stride::Stride;
use crate::tuple::{Profile, Tuple};

/// A shape and a congruent stride.
///
/// Every public constructor checks its inputs through [`Shape`] and [`Stride`], and the algebra
/// derives new layouts only from checked ones. Every `Layout` therefore has positive extents
/// and strides in one codomain.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    /// The domain: the layout accepts the coordinates of this shape.
    shape: Shape,
    /// The codomain step of each leaf of `shape`, congruent to it.
    stride: Stride,
}

impl Layout {
    /// Builds `shape:stride`. A stride leaf opposite a subshape starts that subshape's compact
    /// strides, so shape `(4, 3)` with stride `2` is `(4, 3):(2, 8)`.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Incongruent`] when the stride is not weakly congruent to the shape.
    pub fn new(shape: Shape, stride: Stride) -> Result<Layout, LayoutError> {
        // PERF: a congruent stride, the usual input, needs no expansion or copy.
        if shape.as_tuple().congruent(stride.as_tuple()) {
            return Ok(Layout { shape, stride });
        }
        let expanded = expand(stride.as_tuple(), &shape).ok_or_else(|| {
            LayoutError::Incongruent { shape: shape.to_string(), stride: stride.to_string() }
        })?;
        Ok(Layout { shape, stride: expanded })
    }

    /// Builds a layout from a derived shape and a stride already congruent to it.
    pub(crate) fn from_parts(shape: Shape, stride: Stride) -> Layout {
        debug_assert!(
            shape.as_tuple().congruent(stride.as_tuple()),
            "{shape} and {stride} are congruent"
        );
        Layout { shape, stride }
    }

    /// Builds a layout from a derived shape and a weakly congruent stride, as [`Layout::new`]
    /// expands it.
    pub(crate) fn from_expanded(shape: Shape, stride: &Tuple<Offset>) -> Layout {
        let stride = expand(stride, &shape).expect("derived strides are weakly congruent");
        Layout { shape, stride }
    }

    /// Returns the concatenation of `modes` as the top-level modes of one layout (Equation 11).
    pub fn from_modes(modes: impl IntoIterator<Item = Layout>) -> Layout {
        let (shapes, strides): (Vec<_>, Vec<_>) = modes
            .into_iter()
            .map(|mode| (mode.shape.as_tuple().clone(), mode.stride.as_tuple().clone()))
            .unzip();
        let shape = Shape::from_derived(Tuple::Node(shapes));
        Layout { shape, stride: Stride::from_derived(Tuple::Node(strides)) }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &Stride {
        &self.stride
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn depth(&self) -> usize {
        self.shape.depth()
    }

    pub fn size(&self) -> Int {
        self.shape.size()
    }

    /// Returns the top-level modes. A leaf layout is its own only mode.
    pub fn modes(&self) -> impl Iterator<Item = Layout> + '_ {
        self.shape.modes().zip(self.stride.modes()).map(|(shape, stride)| Layout { shape, stride })
    }

    /// Returns the sublayout at `path`, one mode index per level.
    pub fn get(&self, path: &[usize]) -> Option<Layout> {
        let shape = self.shape.as_tuple().get(path)?.clone();
        let stride = self.stride.as_tuple().get(path)?.clone();
        Some(Layout { shape: Shape::from_derived(shape), stride: Stride::from_derived(stride) })
    }

    /// Returns the offset of any coordinate of the shape: an index, an n-D coordinate, or the
    /// natural coordinate.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Coordinate`] when `coord` is not weakly congruent to the shape.
    pub fn call(&self, coord: &Coord) -> Result<Offset, LayoutError> {
        let natural = self.shape.idx2crd(coord)?;
        Ok(self.stride.inner_product(&natural))
    }

    /// Returns the offset of a 1-D index. Every index is a coordinate of every shape.
    pub fn at(&self, index: impl Into<Int>) -> Offset {
        self.call(&Tuple::Leaf(index.into())).expect("an index is a coordinate of every shape")
    }

    /// Returns the offset of another layout's codomain element read as a coordinate. An
    /// arithmetic tuple is padded with zeros to this layout's rank.
    pub(crate) fn call_offset(&self, offset: &Offset) -> Result<Offset, LayoutError> {
        match offset.as_tuple() {
            Tuple::Leaf(_) => self.call(offset.as_tuple()),
            Tuple::Node(components) => {
                let mut coord = components.clone();
                coord.resize(self.rank().max(coord.len()), Tuple::Leaf(Int::Static(0)));
                self.call(&Tuple::Node(coord))
            }
        }
    }

    /// Splits `coord` into the offset of its fixed leaves and the layout of its free ones
    /// (Equation 10). A free leaf is `None`, and the subshape opposite each free leaf becomes one
    /// mode of the result, in leaf order.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Coordinate`] when `coord` is not weakly congruent to the shape.
    pub fn slice(&self, coord: &Tuple<Option<Int>>) -> Result<(Offset, Layout), LayoutError> {
        // Free leaves contribute nothing to the offset, which is the layout at coordinate 0
        // along them.
        let fixed = coord.map(&mut |leaf| leaf.clone().unwrap_or(Int::Static(0)));
        let offset = self.call(&fixed)?;

        let mut free = Vec::new();
        collect_free(coord, self, &mut free)?;
        Ok((offset, Layout::from_modes(free)))
    }

    /// Returns the profile of the codomain: a leaf for Z, one leaf per axis of Z^n.
    pub fn coprofile(&self) -> Profile {
        self.stride.coprofile()
    }

    /// Returns one past the largest offset along each codomain axis.
    pub fn coshape(&self) -> Tuple<Int> {
        let extents = self.shape.extents();
        let last = extents
            .into_iter()
            .zip(self.stride.steps())
            .fold(Offset::zero(), |sum, (s, d)| sum.add(&d.scale(&(s - &Int::Static(1)))));
        last.as_tuple().map(&mut |end| end + &Int::Static(1))
    }

    /// Returns the layout with a concrete value substituted for every launch parameter.
    #[must_use]
    pub fn eval(&self, value: &impl Fn(&Param) -> i64) -> Layout {
        let shape = self.shape.as_tuple().map(&mut |s| Int::Static(s.eval(value)));
        let stride = self.stride.as_tuple().map(&mut |d| d.eval(value));
        Layout { shape: Shape::from_derived(shape), stride: Stride::from_derived(stride) }
    }
}

/// Collects the sublayouts opposite the free leaves of `coord`, in leaf order.
fn collect_free(
    coord: &Tuple<Option<Int>>,
    layout: &Layout,
    free: &mut Vec<Layout>,
) -> Result<(), LayoutError> {
    match coord {
        Tuple::Leaf(None) => free.push(layout.clone()),
        Tuple::Leaf(Some(_)) => {}
        Tuple::Node(coords) if coords.len() == layout.rank() && layout.depth() > 0 => {
            for (c, mode) in coords.iter().zip(layout.modes()) {
                collect_free(c, &mode, free)?;
            }
        }
        Tuple::Node(_) => {
            return Err(LayoutError::Coordinate {
                coord: format!("{coord:?}"),
                shape: layout.shape.to_string(),
            });
        }
    }
    Ok(())
}

/// Expands each stride leaf over the subshape opposite it into that subshape's compact strides.
fn expand(stride: &Tuple<Offset>, shape: &Shape) -> Option<Stride> {
    match (stride, shape.as_tuple()) {
        (Tuple::Leaf(start), _) => Some(shape.compact_stride(start)),
        (Tuple::Node(steps), Tuple::Node(modes)) if steps.len() == modes.len() => {
            let expanded = steps.iter().zip(shape.modes()).map(|(d, mode)| expand(d, &mode));
            let expanded: Option<Vec<Stride>> = expanded.collect();
            let expanded = expanded?.into_iter().map(|stride| stride.as_tuple().clone());
            Some(Stride::from_derived(Tuple::node(expanded)))
        }
        _ => None,
    }
}

/// The compact column-major layout of `shape`, as PyCuTe's `Layout(shape)`.
impl From<Shape> for Layout {
    fn from(shape: Shape) -> Self {
        let stride = shape.compact_stride(&Offset::from(1));
        Layout { shape, stride }
    }
}

/// Nests layouts as the tuple does: a node becomes a layout whose modes are its children.
impl From<Tuple<Layout>> for Layout {
    fn from(layouts: Tuple<Layout>) -> Self {
        match layouts {
            Tuple::Leaf(layout) => layout,
            Tuple::Node(modes) => Layout::from_modes(modes.into_iter().map(Layout::from)),
        }
    }
}
