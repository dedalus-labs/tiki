// Copyright © 2026 Dedalus Labs, Inc.

//! Hierarchical tuples, the one recursive structure shared by shapes, strides, coordinates and
//! codomain offsets (Cecka, Definition 2.2). Every other module is written as recursion over it.

use std::fmt;

/// A leaf, or an ordered sequence of hierarchical tuples.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tuple<T> {
    /// A single value.
    Leaf(T),
    /// An ordered sequence of modes, possibly empty.
    Node(Vec<Tuple<T>>),
}

/// The structure of a tuple without its values. A leaf profile stands for a whole subtree.
pub type Profile = Tuple<()>;

impl<T> Tuple<T> {
    pub fn node(modes: impl IntoIterator<Item = Tuple<T>>) -> Self {
        Tuple::Node(modes.into_iter().collect())
    }

    /// The top-level modes. A leaf is its own only mode, so every tuple has a rank.
    pub fn modes(&self) -> &[Tuple<T>] {
        match self {
            Tuple::Leaf(_) => std::slice::from_ref(self),
            Tuple::Node(modes) => modes,
        }
    }

    pub fn into_modes(self) -> Vec<Tuple<T>> {
        match self {
            Tuple::Leaf(_) => vec![self],
            Tuple::Node(modes) => modes,
        }
    }

    pub fn rank(&self) -> usize {
        self.modes().len()
    }

    pub fn depth(&self) -> usize {
        match self {
            Tuple::Leaf(_) => 0,
            Tuple::Node(modes) => 1 + modes.iter().map(Tuple::depth).max().unwrap_or(0),
        }
    }

    /// The subtree at `path`, one mode index per level.
    pub fn get(&self, path: &[usize]) -> Option<&Tuple<T>> {
        match path.split_first() {
            None => Some(self),
            Some((&mode, rest)) => self.modes().get(mode)?.get(rest),
        }
    }

    /// Leaves in colexicographic mode order, which is left to right.
    pub fn leaves(&self) -> Vec<&T> {
        let mut leaves = Vec::new();
        self.visit(&mut |leaf| leaves.push(leaf));
        leaves
    }

    fn visit<'a>(&'a self, f: &mut impl FnMut(&'a T)) {
        match self {
            Tuple::Leaf(leaf) => f(leaf),
            Tuple::Node(modes) => modes.iter().for_each(|mode| mode.visit(f)),
        }
    }

    pub fn map<U>(&self, f: &mut impl FnMut(&T) -> U) -> Tuple<U> {
        match self {
            Tuple::Leaf(leaf) => Tuple::Leaf(f(leaf)),
            Tuple::Node(modes) => Tuple::node(modes.iter().map(|mode| mode.map(f))),
        }
    }

    pub fn profile(&self) -> Profile {
        self.map(&mut |_| ())
    }

    /// Pairs each leaf of `self` with the subtree of `other` in the same position. Defined exactly
    /// when `self` is weakly congruent to `other` (Definition 2.4), and `None` otherwise.
    pub fn zip<U, V>(
        &self,
        other: &Tuple<U>,
        f: &mut impl FnMut(&T, &Tuple<U>) -> V,
    ) -> Option<Tuple<V>> {
        match (self, other) {
            (Tuple::Leaf(leaf), _) => Some(Tuple::Leaf(f(leaf, other))),
            (Tuple::Node(ours), Tuple::Node(theirs)) if ours.len() == theirs.len() => {
                let modes = ours.iter().zip(theirs).map(|(a, b)| a.zip(b, f));
                modes.collect::<Option<Vec<_>>>().map(Tuple::Node)
            }
            _ => None,
        }
    }

    /// Same profile, leaf for leaf (Definition 2.3).
    pub fn congruent<U>(&self, other: &Tuple<U>) -> bool {
        match (self, other) {
            (Tuple::Leaf(_), Tuple::Leaf(_)) => true,
            (Tuple::Node(ours), Tuple::Node(theirs)) => {
                ours.len() == theirs.len() && ours.iter().zip(theirs).all(|(a, b)| a.congruent(b))
            }
            _ => false,
        }
    }

    pub fn weakly_congruent<U>(&self, other: &Tuple<U>) -> bool {
        self.zip(other, &mut |_, _| ()).is_some()
    }

    /// Rebuilds `profile` with values taken in leaf order.
    pub fn from_leaves<U>(values: &mut impl Iterator<Item = T>, profile: &Tuple<U>) -> Self {
        match profile {
            Tuple::Leaf(_) => Tuple::Leaf(values.next().expect("one value per profile leaf")),
            Tuple::Node(modes) => {
                Tuple::node(modes.iter().map(|mode| Tuple::from_leaves(values, mode)))
            }
        }
    }

    /// Returns the path of every leaf, in leaf order.
    pub fn leaf_paths(&self) -> Vec<Vec<usize>> {
        match self {
            Tuple::Leaf(_) => vec![Vec::new()],
            Tuple::Node(modes) => {
                let mut paths = Vec::new();
                for (i, mode) in modes.iter().enumerate() {
                    for mut path in mode.leaf_paths() {
                        path.insert(0, i);
                        paths.push(path);
                    }
                }
                paths
            }
        }
    }

    /// Strips rank-1 nodes, so `((4,),)` becomes `4`.
    #[must_use]
    pub fn unwrap(self) -> Self {
        match self {
            Tuple::Node(mut modes) if modes.len() == 1 => modes.pop().expect("rank 1").unwrap(),
            other => other,
        }
    }
}

/// Prints as Python prints a tuple: `(4, (2, 3))`, and `(4,)` for one mode.
impl<T: fmt::Display> fmt::Display for Tuple<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let modes = match self {
            Tuple::Leaf(leaf) => return write!(f, "{leaf}"),
            Tuple::Node(modes) => modes,
        };
        write!(f, "(")?;
        for (i, mode) in modes.iter().enumerate() {
            let separator = if i == 0 { "" } else { ", " };
            write!(f, "{separator}{mode}")?;
        }
        let one_mode = if modes.len() == 1 { "," } else { "" };
        write!(f, "{one_mode})")
    }
}

impl<T> From<T> for Tuple<T> {
    fn from(leaf: T) -> Self {
        Tuple::Leaf(leaf)
    }
}
