.. _optimizers:

.. currentmodule:: tiki.optimizers

Optimizers
==========

The optimizers in Tiki can be used both with :mod:`tiki.nn` but also with pure
:mod:`tiki` functions. A typical example involves calling
:meth:`Optimizer.update` to update a model's parameters based on the loss
gradients and subsequently calling :func:`tiki.eval` to evaluate both the
model's parameters and the **optimizer state**.

.. code-block:: python

    # Create a model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    tk.eval(model.parameters())

    # Create the gradient function and the optimizer
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    for e in range(num_epochs):
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)

            # Update the model with the gradients. So far no computation has happened.
            optimizer.update(model, grads)

            # Compute the new parameters but also the optimizer state.
            tk.eval(model.parameters(), optimizer.state)

Saving and Loading
------------------

To serialize an optimizer, save its state. To load an optimizer, load and set
the saved state. Here's a simple example:

.. code-block:: python

   import tiki as tk
   from tiki.utils import tree_flatten, tree_unflatten
   import tiki.optimizers as optim

   optimizer = optim.Adam(learning_rate=1e-2)

   # Perform some updates with the optimizer
   model = {"w" : tk.zeros((5, 5))}
   grads = {"w" : tk.ones((5, 5))}
   optimizer.update(model, grads)

   # Save the state
   state = tree_flatten(optimizer.state, destination={})
   tk.save_safetensors("optimizer.safetensors", state)

   # Later on, for example when loading from a checkpoint,
   # recreate the optimizer and load the state
   optimizer = optim.Adam(learning_rate=1e-2)

   state = tree_unflatten(tk.load("optimizer.safetensors"))
   optimizer.state = state

Note, not every optimizer configuation parameter is saved in the state. For
example, for Adam the learning rate is saved but the ``betas`` and ``eps``
parameters are not. A good rule of thumb is if the parameter can be scheduled
then it will be included in the optimizer state.

.. toctree::

   optimizers/optimizer
   optimizers/common_optimizers
   optimizers/schedulers

.. autosummary::
   :toctree: _autosummary

   clip_grad_norm
