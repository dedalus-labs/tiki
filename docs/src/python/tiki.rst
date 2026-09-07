.. _tiki-layout-api:

Tiki layout API
===============

.. currentmodule:: mlx.tiki

See :ref:`tiki-layouts` for the model and :ref:`tiki-layout-recipes` for examples.

.. py:data:: Coordinate
   :module: mlx.tiki.composed

   An integer coordinate, a nested tuple of coordinates, or a slice marker.
   ``None`` and ``slice(None)`` retain a mode during slicing.

Maps and transforms
-------------------

.. autoclass:: Layout
   :members: swizzle

.. autoclass:: Swizzle
   :special-members: __call__

.. autoclass:: ComposedLayout
   :members: swizzle
   :special-members: __call__

.. autofunction:: compose

.. autofunction:: slice_and_offset

.. autoclass:: LayoutError

Storage boundaries
------------------

.. autoclass:: ArrayEngine

.. autofunction:: from_array

.. autofunction:: realize

Domain operations
-----------------

.. autofunction:: unsqueeze

.. autofunction:: squeeze

.. autofunction:: expand

``logical_divide``, ``zipped_divide``, ``coalesce``, ``complement``,
``logical_product``, ``blocked_product``, ``raked_product``, ``left_inverse``,
``right_inverse``, ``nullspace``, and ``recast`` expose the vendored PyCuTe
algebra through the same namespace. They do not imply general support for
nonlinear operands. Use composition to apply a transform to a derived affine
domain.
