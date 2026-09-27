.. _nn_distributed:

Distributed
-----------

Helper Routines
^^^^^^^^^^^^^^^

The :code:`tiki.nn.layers.distributed` package contains helpful routines to 
create sharded layers from existing :class:`Modules <tiki.nn.Module>`.

.. currentmodule:: tiki.nn.layers.distributed
.. autosummary::
   :toctree: _autosummary

   shard_linear
   shard_inplace
   fully_shard

Layers
^^^^^^

.. currentmodule:: tiki.nn
.. autosummary::
   :toctree: _autosummary
   :template: nn-module-template.rst

   AllToShardedLinear
   ShardedToAllLinear
   QuantizedAllToShardedLinear
   QuantizedShardedToAllLinear
   FullyShardedModule
