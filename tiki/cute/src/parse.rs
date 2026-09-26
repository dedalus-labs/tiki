// Copyright © 2026 Dedalus Labs, Inc.

//! Parsing CuTe notation, the inverse of [`Layout::cute`] and of PyCuTe's `str`.
//!
//! ```text
//! item   := "_" | group [":" group]        a hole, a shape or tuple, or a layout
//! group  := "(" item ("," item)* [","] ")" | sum
//! sum    := term ("+" term)*               an integer, or an arithmetic tuple
//! term   := factor ("*" factor)* ("@" digits)*
//! factor := digits | name | "-" factor     a name is a positive launch parameter
//! ```
//!
//! `5@2@1` is `5 * E(1, 2)`, as PyCuTe prints it: the basis modes are written innermost first.
//!
//! ```
//! use tiki_cute::{Layout, Tiler};
//!
//! let layout: Layout = "(4, (2, N)):(1@0, (1@1, 2@1))".parse()?;
//! assert_eq!(layout.rank(), 2);
//! let tiler: Tiler = "(2:3, _)".parse()?;
//! assert!(matches!(tiler, Tiler::Modes(ref modes) if modes[1].is_none()));
//! # Ok::<(), tiki_cute::LayoutError>(())
//! ```

use crate::LayoutError;
use crate::int::Int;
use crate::layout::Layout;
use crate::offset::Offset;
use crate::param::Param;
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tiler::Tiler;
use crate::tuple::Tuple;
use std::str::FromStr;

/// One parsed item, before the caller says what it must be.
enum Item {
    /// `_`, a tiler mode left as it is.
    Hole,
    /// A leaf value: an integer or an arithmetic tuple.
    Leaf(Offset),
    /// A parenthesized tuple of items.
    Group(Vec<Item>),
    /// `shape:stride`.
    Layout(Layout),
}

/// A cursor over the text being parsed.
struct Parser<'a> {
    /// The whole input, kept for error messages.
    text: &'a str,
    /// Byte offset of the next unread character.
    at: usize,
}

impl FromStr for Layout {
    type Err = LayoutError;

    fn from_str(text: &str) -> Result<Layout, LayoutError> {
        match Parser::parse(text)? {
            Item::Layout(layout) => Ok(layout),
            _ => Err(Parser { text, at: 0 }.error("shape:stride")),
        }
    }
}

impl FromStr for Shape {
    type Err = LayoutError;

    fn from_str(text: &str) -> Result<Shape, LayoutError> {
        Shape::try_from(text.parse::<Tuple<Int>>()?)
    }
}

/// Parses a tuple of integers of any sign, such as a mode order or a recast scale.
impl FromStr for Tuple<Int> {
    type Err = LayoutError;

    fn from_str(text: &str) -> Result<Tuple<Int>, LayoutError> {
        integers(Parser::parse(text)?, text)
    }
}

impl FromStr for Tiler {
    type Err = LayoutError;

    fn from_str(text: &str) -> Result<Tiler, LayoutError> {
        tiler(Parser::parse(text)?, text)
    }
}

impl<'a> Parser<'a> {
    /// Parses all of `text` as one item.
    fn parse(text: &'a str) -> Result<Item, LayoutError> {
        let mut parser = Parser { text, at: 0 };
        let item = parser.item()?;
        parser.skip_space();
        if parser.at < text.len() {
            return Err(parser.error("end of input"));
        }
        Ok(item)
    }

    fn item(&mut self) -> Result<Item, LayoutError> {
        if self.eat('_') {
            return Ok(Item::Hole);
        }
        let group = self.group()?;
        if !self.eat(':') {
            return Ok(group);
        }
        let at = self.at;
        let stride = self.group()?;
        let shape = Shape::try_from(integers(group, self.text)?)?;
        let stride = Stride::try_from(offsets(stride).ok_or_else(|| self.error_at(at, "stride"))?)?;
        Ok(Item::Layout(Layout::new(shape, stride)?))
    }

    fn group(&mut self) -> Result<Item, LayoutError> {
        if !self.eat('(') {
            return Ok(Item::Leaf(self.sum()?));
        }
        if self.eat(')') {
            return Ok(Item::Group(Vec::new()));
        }
        let mut items = vec![self.item()?];
        while self.eat(',') {
            if self.peek() == Some(')') {
                break;
            }
            items.push(self.item()?);
        }
        self.expect(')')?;
        Ok(Item::Group(items))
    }

    fn sum(&mut self) -> Result<Offset, LayoutError> {
        let mut sum = self.term()?;
        while self.eat('+') {
            let at = self.at;
            let term = self.term()?;
            sum = sum
                .checked_add(&term)
                .ok_or_else(|| self.error_at(at, "a term in the same codomain"))?;
        }
        Ok(sum)
    }

    fn term(&mut self) -> Result<Offset, LayoutError> {
        let mut value = self.factor()?;
        while self.eat('*') {
            value = &value * &self.factor()?;
        }
        // PyCuTe writes the innermost basis mode first, so `5@2@1` is `5 * E(1, 2)`.
        let mut path = Vec::new();
        while self.eat('@') {
            let at = self.at;
            let mode =
                usize::try_from(self.digits()?).map_err(|_| self.error_at(at, "a mode index"))?;
            path.insert(0, mode);
        }
        Ok(Offset::scaled_basis(value, &path))
    }

    fn factor(&mut self) -> Result<Int, LayoutError> {
        self.skip_space();
        if self.eat('-') {
            return Ok(-&self.factor()?);
        }
        match self.peek() {
            Some(c) if c.is_ascii_digit() => Ok(Int::Static(self.digits()?)),
            Some(c) if c.is_ascii_alphabetic() => {
                let start = self.at;
                while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
                    self.at += 1;
                }
                Ok(Int::from(Param::positive(&self.text[start..self.at])))
            }
            _ => Err(self.error("an integer or a parameter name")),
        }
    }

    fn digits(&mut self) -> Result<i64, LayoutError> {
        self.skip_space();
        let start = self.at;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.at += 1;
        }
        self.text[start..self.at].parse().map_err(|_| self.error_at(start, "an integer"))
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_space();
        self.text[self.at..].chars().next()
    }

    fn eat(&mut self, expected: char) -> bool {
        let found = self.peek() == Some(expected);
        if found {
            self.at += expected.len_utf8();
        }
        found
    }

    fn expect(&mut self, expected: char) -> Result<(), LayoutError> {
        if self.eat(expected) { Ok(()) } else { Err(self.error("a closing parenthesis")) }
    }

    fn skip_space(&mut self) {
        while self.text[self.at..].starts_with(char::is_whitespace) {
            self.at += 1;
        }
    }

    fn error(&self, expected: &'static str) -> LayoutError {
        self.error_at(self.at, expected)
    }

    fn error_at(&self, at: usize, expected: &'static str) -> LayoutError {
        LayoutError::Parse { text: self.text.into(), at, expected }
    }
}

/// Reads an item as a tuple of integers, the form of a shape.
fn integers(item: Item, text: &str) -> Result<Tuple<Int>, LayoutError> {
    let error = || LayoutError::Parse { text: text.into(), at: 0, expected: "a shape" };
    match item {
        Item::Leaf(offset) => offset.as_int().cloned().map(Tuple::Leaf).ok_or_else(error),
        Item::Group(items) => {
            let modes = items.into_iter().map(|item| integers(item, text));
            Ok(Tuple::Node(modes.collect::<Result<_, _>>()?))
        }
        Item::Hole | Item::Layout(_) => Err(error()),
    }
}

/// Reads an item as a tuple of offsets, the form of a stride.
fn offsets(item: Item) -> Option<Tuple<Offset>> {
    match item {
        Item::Leaf(offset) => Some(Tuple::Leaf(offset)),
        Item::Group(items) => {
            items.into_iter().map(offsets).collect::<Option<_>>().map(Tuple::Node)
        }
        Item::Hole | Item::Layout(_) => None,
    }
}

/// Reads an item as a tiler: a layout, an extent `n` meaning `n:1`, or a tuple of tilers and
/// holes.
fn tiler(item: Item, text: &str) -> Result<Tiler, LayoutError> {
    match item {
        Item::Layout(layout) => Ok(Tiler::Layout(layout)),
        Item::Leaf(offset) => match offset.as_int() {
            Some(extent) => Tiler::try_from(extent.clone()),
            None => Err(LayoutError::Parse { text: text.into(), at: 0, expected: "an extent" }),
        },
        Item::Group(items) => {
            let modes = items.into_iter().map(|item| match item {
                Item::Hole => Ok(None),
                item => tiler(item, text).map(Some),
            });
            Ok(Tiler::Modes(modes.collect::<Result<_, _>>()?))
        }
        Item::Hole => Err(LayoutError::Parse { text: text.into(), at: 0, expected: "a tiler" }),
    }
}
