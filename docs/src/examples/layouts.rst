.. _tiki-layout-recipes:

Layout recipes
==============

These examples run against the public Tiki API. They use the same Engine and
tensor types for affine and swizzled layouts. Read :ref:`tiki-layouts` for the
coordinate and offset conventions.

One Engine, two layouts
-----------------------

.. doctest:: tiki-recipes

   >>> import mlx.core as mx
   >>> import mlx.tiki as tk
   >>> engine = tk.ArrayEngine(mx.arange(16))
   >>> base = tk.Layout((4, 4), (4, 1))
   >>> swizzle = tk.Swizzle(2, 0, 2)
   >>> ordinary = tk.Tensor(engine, base)
   >>> permuted = tk.Tensor(engine, base.swizzle(swizzle))
   >>> ordinary[1, 2], permuted[1, 2]
   (6, 7)
   >>> ordinary.accessor is permuted.accessor
   True
   >>> type(ordinary) is type(permuted)
   True
   >>> [permuted[1, None][column] for column in range(4)]
   [5, 4, 7, 6]

Construction changes the coordinate map, not the allocation or its ownership.
Scalar reads use the Engine and can synchronize.

Tile the domain before composing
--------------------------------

``logical_divide`` factors a domain into tile coordinates and tile indices.
Compose the resulting domain with the Swizzle to preserve the address map.

.. doctest:: tiki-recipes

   >>> divided = tk.logical_divide(tk.Layout(16, 1), tk.Layout(4, 1))
   >>> divided.shape
   (4, 4)
   >>> tiled = divided.swizzle(swizzle)
   >>> tiled(2, 1), swizzle(divided(2, 1))
   (7, 7)
   >>> coalesced = tk.coalesce(tk.Layout((2, 8), (1, 2)))
   >>> coalesced.swizzle(swizzle)(6)
   7

Affine algebra results retain the convenience method. Nonlinear composition
does not require a second tensor API or an ordinary-stride approximation.

Nested transforms
-----------------

Order matters. Another ``swizzle`` call wraps the existing expression rather
than replacing it.

.. doctest:: tiki-recipes

   >>> second = tk.Swizzle(1, 0, 1)
   >>> nested = base.swizzle(swizzle).swizzle(second)
   >>> nested(1, 2), second(swizzle(base(1, 2)))
   (6, 6)
   >>> restored = base.swizzle(swizzle).swizzle(swizzle)
   >>> all(
   ...     restored(row, column) == base(row, column)
   ...     for row in range(4) for column in range(4)
   ... )
   True

Applying an affine outer map to a swizzled inner map also uses composition.

.. doctest:: tiki-recipes

   >>> doubled = tk.compose(tk.Layout(16, 2), base.swizzle(swizzle))
   >>> doubled(1, 2)
   14

Hierarchy and broadcasting
--------------------------

``unsqueeze``, ``squeeze``, and ``expand`` operate on top-level modes. Nested
modes remain nested. Expanding a singleton mode gives it stride zero.

.. doctest:: tiki-recipes

   >>> blocked = tk.Layout(((2, 3), 4), ((1, 2), 6))
   >>> tensor = tk.Tensor(tk.ArrayEngine(mx.arange(24)), blocked)
   >>> inserted = tk.unsqueeze(tensor, 2)
   >>> inserted.layout.shape
   ((2, 3), 4, 1)
   >>> expanded = tk.expand(inserted, (6, 4, 2))
   >>> expanded.layout.shape, expanded.layout.stride
   (((2, 3), 4, 2), ((1, 2), 6, 0))
   >>> expanded[1, 2, 0], expanded[1, 2, 1]
   (13, 13)

These operations currently require a stride layout. For a swizzled tensor,
transform the affine domain first and then compose the Swizzle.

Reverse a retained buffer
-------------------------

Negative strides require an Engine displacement that keeps every address in
bounds. ``realize`` checks the addressed interval before constructing the view.

.. doctest:: tiki-recipes

   >>> reverse = tk.Tensor(tk.ArrayEngine(mx.arange(4), offset=3), tk.Layout(4, -1))
   >>> tk.realize(reverse).tolist()
   [3, 2, 1, 0]
   >>> array = mx.arange(8)[::2]
   >>> tk.realize(tk.from_array(array)).tolist()
   [0, 2, 4, 6]

The second round trip can copy because ``from_array`` normalizes its input.

Invalid maps fail explicitly
----------------------------

Overlapping source and destination fields can destroy information. The Rust
constructor rejects them.

.. doctest:: tiki-recipes

   >>> try:
   ...     tk.Swizzle(2, 0, 1)
   ... except tk.LayoutError as error:
   ...     print(error)
   swizzle fields overlap: abs(shift) must be at least bits, got bits=2, shift=1
   >>> try:
   ...     swizzle(-1)
   ... except tk.LayoutError as error:
   ...     print(error)
   swizzle index must be nonnegative, got -1

A nonlinear layout cannot pass through an ordinary-stride boundary.

.. doctest:: tiki-recipes

   >>> try:
   ...     permuted.layout.stride
   ... except tk.LayoutError as error:
   ...     print(error)
   a composed layout has no stride. Require an affine layout
   >>> try:
   ...     tk.realize(permuted)
   ... except tk.LayoutError as error:
   ...     print(error)
   cannot realize a composed layout as a view: SW_2_0_2 o {0} o (4, 4):(4, 1)

The same boundary rejects XOR-valued stride scalars. An ``F2`` stride adds with
XOR, not integer addition. Coercing it to ``int`` changes the coordinate map.

.. doctest:: tiki-recipes

   >>> xor_layout = tk.Layout((2, 2), (tk.F2(1), tk.F2(1)))
   >>> try:
   ...     tk.realize(tk.Tensor(engine, xor_layout))
   ... except tk.LayoutError as error:
   ...     print(error)
   realize requires integer strides, not coordinate or XOR strides
