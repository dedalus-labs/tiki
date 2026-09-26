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
use crate::xor::Xor;

/// A shape and a congruent stride.
///
/// Every public constructor checks its inputs through [`Shape`] and [`Stride`], and the algebra
/// derives new layouts only from checked ones. Every `Layout` therefore has positive extents
/// and strides in one codomain. A layout with XOR strides also has static extents, because a
/// carry-less product needs every bit of the extent it multiplies.
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
    /// Returns [`LayoutError::Incongruent`] when the stride is not weakly congruent to the shape,
    /// and [`LayoutError::XorOperand`] for XOR strides beside a dynamic extent.
    pub fn new(shape: Shape, stride: Stride) -> Result<Layout, LayoutError> {
        static_under_xor(&shape, &stride)?;
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
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::MixedCodomain`] when the modes' strides lie in codomains that have
    /// no common sum, such as `1` beside `1@0`, and [`LayoutError::XorOperand`] when one mode's
    /// XOR strides meet another mode's dynamic extent.
    pub fn try_from_modes(modes: impl IntoIterator<Item = Layout>) -> Result<Layout, LayoutError> {
        let layout = Layout::from_modes(modes);
        Stride::try_from(layout.stride.as_tuple().clone())?;
        static_under_xor(&layout.shape, &layout.stride)?;
        Ok(layout)
    }

    /// Concatenates modes the algebra derived in one codomain, without checking it again.
    pub(crate) fn from_modes(modes: impl IntoIterator<Item = Layout>) -> Layout {
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
    /// Returns [`LayoutError::Coordinate`] when `coord` is not weakly congruent to the shape, and
    /// [`LayoutError::XorOperand`] when a coordinate leaf opposite XOR strides is negative or
    /// dynamic.
    pub fn call(&self, coord: &Coord) -> Result<Offset, LayoutError> {
        let natural = self.shape.idx2crd(coord)?;
        if self.stride.is_xor() {
            for leaf in natural.leaves() {
                Xor::try_from(leaf)?;
            }
        }
        Ok(self.stride.inner_product(&natural))
    }

    /// Returns the offset of a 1-D index. Every index is a coordinate of every shape.
    ///
    /// # Panics
    ///
    /// Panics when this layout has XOR strides and `index` is negative or dynamic, which
    /// [`Layout::call`] refuses instead.
    pub fn at(&self, index: impl Into<Int>) -> Offset {
        self.call(&Tuple::Leaf(index.into()))
            .unwrap_or_else(|error| panic!("an index is a coordinate of every shape: {error}"))
    }

    /// Returns the offset of another layout's codomain element read as a coordinate, so a
    /// layout can be evaluated at the result of another.
    ///
    /// An integer is an index and an arithmetic tuple is a coordinate, padded with zeros to this
    /// layout's rank. An XOR value is an XOR index: [`Shape::idx2crd_xor`] splits it over the
    /// shape carry-lessly, and each XOR coordinate multiplies its stride carry-lessly, so the
    /// result is an XOR value, as PyCuTe evaluates a layout at an `F2` index.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Layout::call`] and [`Shape::idx2crd_xor`], and for an XOR index
    /// [`LayoutError::XorStride`] or [`LayoutError::XorOperand`] when a stride has no carry-less
    /// product with a coordinate.
    pub fn call_offset(&self, offset: &Offset) -> Result<Offset, LayoutError> {
        if let Some(index) = offset.as_xor() {
            let natural = self.shape.idx2crd_xor(index)?;
            let mut sum = Offset::zero();
            for (c, d) in natural.leaves().into_iter().zip(self.stride.steps()) {
                sum = sum.add(&d.scale_xor(*c, "evaluation")?);
            }
            return Ok(sum);
        }
        match offset.as_tuple().expect("an offset that is not XOR is a tuple") {
            index @ Tuple::Leaf(_) => self.call(index),
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

    /// Returns the profile of the codomain: a leaf for Z and for the XOR codomain, one leaf per
    /// axis of Z^n.
    pub fn coprofile(&self) -> Profile {
        self.stride.coprofile()
    }

    /// Returns the offset of the last coordinate plus one along each codomain axis, PyCuTe's
    /// `coshape`. With nonnegative strides this is one past the largest offset.
    ///
    /// XOR cancels bits instead of accumulating them, so for XOR strides the bound is the
    /// smallest power of two above every mode's largest contribution `(s - 1) * d`, as in PyCuTe.
    pub fn coshape(&self) -> Tuple<Int> {
        if self.stride.is_xor() {
            let width = self.shape.extents().into_iter().zip(self.stride.steps()).fold(
                0,
                |width, (s, d)| {
                    let reach = d.scale(&(s - &Int::Static(1)));
                    width.max(reach.as_xor().map_or(0, |bits| crate::xor::bit_length(bits.value())))
                },
            );
            let bound = i64::try_from(1_u64 << width).expect(crate::poly::OVERFLOW);
            return Tuple::Leaf(Int::Static(bound));
        }
        let extents = self.shape.extents();
        let last = extents
            .into_iter()
            .zip(self.stride.steps())
            .fold(Offset::zero(), |sum, (s, d)| sum.add(&d.scale(&(s - &Int::Static(1)))));
        let last = last.as_tuple().expect("strides that are not XOR sum to a tuple");
        last.map(&mut |end| end + &Int::Static(1))
    }

    /// Returns the layout with a concrete value substituted for every launch parameter.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Inadmissible`] when a value breaks its parameter's facts, the check
    /// the launcher makes before a kernel runs. Every fact the algebra relied on then holds.
    pub fn eval(&self, value: &impl Fn(&Param) -> i64) -> Result<Layout, LayoutError> {
        // Evaluation takes a shared closure, so the first refusal is recorded in a cell.
        let refused = std::cell::RefCell::new(None);
        let checked = |param: &Param| {
            let v = value(param);
            let mut first = refused.borrow_mut();
            if !param.admits(v) && first.is_none() {
                *first = Some(LayoutError::Inadmissible { param: param.to_string(), value: v });
            }
            v
        };
        let shape = self.shape.as_tuple().map(&mut |s| Int::Static(s.eval(&checked)));
        let stride = self.stride.as_tuple().map(&mut |d| d.eval(&checked));
        match refused.into_inner() {
            Some(error) => Err(error),
            None => Ok(Layout {
                shape: Shape::from_derived(shape),
                stride: Stride::from_derived(stride),
            }),
        }
    }
}

/// Checks that every extent opposite XOR strides is static, so every carry-less product the
/// layout forms is defined.
fn static_under_xor(shape: &Shape, stride: &Stride) -> Result<(), LayoutError> {
    if !stride.is_xor() {
        return Ok(());
    }
    match shape.extents().into_iter().find(|extent| !extent.is_static()) {
        Some(extent) => Err(LayoutError::XorOperand { operand: extent.to_string() }),
        None => Ok(()),
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

impl Layout {
    /// Nests layouts the algebra derived as the tuple does: a node becomes a layout whose modes
    /// are its children.
    pub(crate) fn nest(layouts: Tuple<Layout>) -> Layout {
        match layouts {
            Tuple::Leaf(layout) => layout,
            Tuple::Node(modes) => Layout::from_modes(modes.into_iter().map(Layout::nest)),
        }
    }
}
