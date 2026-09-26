// Copyright © 2026 Dedalus Labs, Inc.

//! Products and divisions: tiling one layout by another (Cecka, Sections 3.5 and 3.6).
//!
//! Both are built from composition and complement. A product repeats `A` at the offsets the
//! tiler selects from the complement of `A`. A division splits `A` into the part the tiler
//! selects, the tile, and the rest, which indexes the tiles.

use crate::LayoutError;
use crate::layout::Layout;
use crate::tiler::Tiler;

impl Layout {
    /// Returns `(self, complement(self) ∘ tiler)`: one copy of `self` per coordinate of the
    /// tiler, placed in the offsets `self` leaves free.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Layout::complement`] and [`Layout::compose`].
    pub fn logical_product(&self, tiler: &Tiler) -> Result<Layout, LayoutError> {
        match tiler {
            Tiler::Layout(_) => {
                let copies = self.complement()?.compose(tiler)?;
                Ok(Layout::from_modes([self.clone(), copies]))
            }
            Tiler::Modes(tilers) => {
                self.each_mode("logical product", tilers, Layout::logical_product)
            }
        }
    }

    /// Returns the product that keeps whole copies of `self` together: mode `i` is
    /// `(self_i, copies_i)`, so the first coordinate of each mode walks inside one copy.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::TilerRank`] when the ranks differ, and the errors of
    /// [`Layout::logical_product`].
    pub fn blocked_product(&self, tiler: &Layout) -> Result<Layout, LayoutError> {
        let pairs = self.copies(tiler)?;
        Ok(Layout::from_modes(pairs.map(|(copy, repeat)| Layout::from_modes([copy, repeat]))))
    }

    /// Returns the product that interleaves the copies: mode `i` is `(copies_i, self_i)`, so
    /// neighboring coordinates of each mode fall in different copies of `self`.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::TilerRank`] when the ranks differ, and the errors of
    /// [`Layout::logical_product`].
    pub fn raked_product(&self, tiler: &Layout) -> Result<Layout, LayoutError> {
        let pairs = self.copies(tiler)?;
        Ok(Layout::from_modes(pairs.map(|(copy, repeat)| Layout::from_modes([repeat, copy]))))
    }

    /// Pairs each mode of `self` with the matching mode of `complement(self) ∘ tiler`.
    fn copies(
        &self,
        tiler: &Layout,
    ) -> Result<impl Iterator<Item = (Layout, Layout)>, LayoutError> {
        if self.rank() != tiler.rank() {
            return Err(LayoutError::TilerRank {
                operation: "blocked or raked product",
                tiler: tiler.rank(),
                layout: self.rank(),
            });
        }
        let copies = self.complement()?.compose(&Tiler::from(tiler.clone()))?;
        let copies: Vec<Layout> = copies.modes().collect();
        let modes: Vec<Layout> = self.modes().collect();
        Ok(modes.into_iter().zip(copies))
    }

    /// Returns `self ∘ (tiler, complement(tiler, shape(self)))`: mode 0 is the tile the tiler
    /// selects, and mode 1 indexes the tiles.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Layout::complement_in`] and [`Layout::compose`].
    pub fn logical_divide(&self, tiler: &Tiler) -> Result<Layout, LayoutError> {
        match tiler {
            Tiler::Layout(tile) => {
                let rest = tile.complement_in(self.shape())?;
                self.compose(&Tiler::from(Layout::from_modes([tile.clone(), rest])))
            }
            Tiler::Modes(tilers) => {
                self.each_mode("logical divide", tilers, Layout::logical_divide)
            }
        }
    }

    /// Returns the division with every tile mode gathered in mode 0 and every tile index in
    /// mode 1, so the slice `(_, t)` selects all of tile `t`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Tiler::to_layout`] and [`Layout::logical_divide`].
    pub fn zipped_divide(&self, tiler: &Tiler) -> Result<Layout, LayoutError> {
        self.logical_divide(&Tiler::from(tiler.to_layout()?))
    }
}
