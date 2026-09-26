.. _tiki-layouts:

Layouts and index transforms
============================

Tiki separates storage from indexing. An Engine owns or retains storage. A
layout maps coordinates to element offsets. The same tensor type accepts an
integer-stride layout or a composed layout::

   tensor[coordinate] = engine[layout(coordinate)]

A :class:`mlx.tiki.Swizzle` is a first-class indexing transform. It mixes index
bits with exclusive-or (XOR). It permutes index values, rather than merely
reordering individual bits. Composing it with a layout supplies the coordinate
domain::

   layout(coordinate) = swizzle(offset + base_layout(coordinate))

There is no separate swizzled tensor family. A composed value participates in
the same layout interface, but it generally has no ordinary stride tuple.

.. important::

   This API requires a Tiki build with its Rust indexing extension. Installing
   upstream MLX alone does not provide ``mlx.tiki``. See :ref:`tiki-layout-build`.

Shape, stride, and coordinates
------------------------------

Strides are measured in elements, not bytes. In an integer-stride layout, each
coordinate contributes its coordinate value multiplied by its stride.

.. doctest:: tiki-layouts

   >>> import mlx.tiki as tk
   >>> row_major = tk.Layout((4, 8), (8, 1))
   >>> column_major = tk.Layout((4, 8), (1, 4))
   >>> row_major(2, 3), column_major(2, 3)
   (19, 14)
   >>> tk.Layout((4, 8)) == column_major
   True

The default strides follow CuTe's column-major order. Pass explicit strides
when expressing a row-major buffer. A hierarchical shape retains groups of
modes instead of flattening them into unrelated axes.

.. doctest:: tiki-layouts

   >>> blocked = tk.Layout((3, (2, 4)), (2, (1, 6)))
   >>> blocked.shape
   (3, (2, 4))
   >>> blocked(17), blocked(2, 5), blocked(2, (1, 2))
   (17, 17, 17)

An integral coordinate is decomposed with the first mode varying fastest.
This coordinate convention is independent of the physical stride order.
Nested integer strides can describe blocked maps that a flat stride vector
cannot describe on the same top-level axes. They cannot describe every XOR map.

A swizzle is a value
--------------------

``Swizzle(bits, base, shift)`` describes two disjoint bit fields:

* ``bits`` is the width of each field.
* ``base`` is the number of untouched low bits below both fields.
* ``shift`` is the signed distance from the destination field to the source.
  A positive shift copies high bits toward low bits. A negative shift copies
  low bits toward high bits.

Rust validates the parameters and retains them immutably. Both fields must
fit within bits 0 through 62, and ``abs(shift) >= bits``. Input indices must
fit a nonnegative signed 64-bit integer.

.. doctest:: tiki-layouts

   >>> swizzle = tk.Swizzle(2, 0, 2)
   >>> swizzle.bits, swizzle.base, swizzle.shift
   (2, 0, 2)
   >>> swizzle(6), swizzle(swizzle(6))
   (7, 6)

The second application restores the input. The source field is unchanged by
the first application, and XOR with the same value twice cancels.

Composition supplies the domain
-------------------------------

Use :func:`mlx.tiki.compose` or the layout's ``swizzle`` method. Both construct
the same composition. The transform does not allocate storage or change the
coordinate domain.

.. doctest:: tiki-layouts

   >>> base = tk.Layout((4, 4), (4, 1))
   >>> tiled = tk.compose(swizzle, base)
   >>> tiled == base.swizzle(swizzle)
   True
   >>> tiled.shape
   (4, 4)
   >>> tk.is_layout(swizzle), tk.is_layout(tiled)
   (False, True)
   >>> for row in range(4):
   ...     print([tiled(row, column) for column in range(4)])
   [0, 1, 2, 3]
   [5, 4, 7, 6]
   [10, 11, 8, 9]
   [15, 14, 13, 12]
   >>> str(tiled)
   'SW_2_0_2 o {0} o (4, 4):(4, 1)'

The column increment changes between rows. No fixed pair of integer strides
describes this map on the same ``(4, 4)`` domain. Asking for ``tiled.stride``
raises :class:`mlx.tiki.LayoutError`. An explicit composition remains inspectable
even when a restricted subdomain could admit an affine simplification.

Offsets stay inside the transform
---------------------------------

An internal layout offset and an Engine displacement are different operations.
Moving an offset outside XOR changes the map.

.. doctest:: tiki-layouts

   >>> inner = tk.Layout(4, 1)
   >>> shifted = tk.compose(swizzle, inner, offset=4)
   >>> [shifted(column) for column in range(4)]
   [5, 4, 7, 6]
   >>> [4 + swizzle(inner(column)) for column in range(4)]
   [4, 5, 6, 7]

Slicing retains this distinction. :func:`mlx.tiki.slice_and_offset` returns a
residual layout and an external Engine displacement. For every retained
coordinate, their sum equals the original address.

.. doctest:: tiki-layouts

   >>> residual, displacement = tk.slice_and_offset((1, None), tiled)
   >>> displacement
   0
   >>> [residual(column) for column in range(4)]
   [5, 4, 7, 6]
   >>> all(displacement + residual(column) == tiled(1, column) for column in range(4))
   True

Composition and storage safety
------------------------------

A valid Swizzle is a bijection on its index domain. This alone does not prove
that a particular layout stays inside a particular allocation. Shape, internal
offset, Engine displacement, and backing storage size still matter.

``ArrayEngine`` checks scalar accesses against its retained array. ``realize``
checks integer-affine bounds before constructing an MLX ``as_strided`` view.
It rejects composed or XOR-valued layouts. It does not reinterpret them as
ordinary strides or silently gather them into a dense array.

``from_array`` explicitly normalizes a noncontiguous input and can allocate.
Reference scalar indexing can synchronize with the device. The layout API does
not make a scalar Python loop into a GPU kernel.

Compiler policy
---------------

The experimental CuTe transpose schedule uses this same Rust-backed Swizzle
type. Its 32-by-32 tile imposes an additional ``bits + base <= 5`` and
``shift == 5`` contract. That restriction belongs to the schedule, not to the
general indexing transform.

Explicit user layouts remain inspectable and are not rewritten by this API.
A compiler can select a different layout for private temporary storage without
changing a user-visible array. The current transpose schedule makes that choice
explicit. Automatic hardware-dependent selection is not implemented.

General composed ``mlx.tiki`` tensor layouts are not yet accepted by the
elementwise ``tk.compile`` path, which consumes MLX shape/stride profiles.
The transpose path lowers Swizzle parameters for its private shared-memory
tile. These are different integration claims.

Continue with :ref:`tiki-layout-recipes` for tensor access, tiling, broadcasting,
negative strides, nested composition, and error examples. The
:ref:`tiki-layout-api` lists the public API.

The composition model follows `CuTe's swizzle layout implementation
<https://github.com/NVIDIA/cutlass/blob/7107b05535f8977f5ecb9d01ee203205b1fd9bc4/include/cute/swizzle_layout.hpp>`_.
