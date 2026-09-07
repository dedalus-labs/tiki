.. _data_parallelism:

Data Parallelism
================

Tiki enables efficient data parallel distributed training through its
distributed communication primitives.

.. _training_example:

Training Example
----------------

In this section we will adapt an Tiki training loop to support data parallel
distributed training. Namely, we will average the gradients across a set of
hosts before applying them to the model.

Our training loop looks like the following code snippet if we omit the model,
dataset, and optimizer initialization.

.. code:: python

    model = ...
    optimizer = ...
    dataset = ...

    def step(model, x, y):
        loss, grads = loss_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    for x, y in dataset:
        loss = step(model, x, y)
        tk.eval(loss, model.parameters())

All we have to do to average the gradients across machines is perform an
:func:`tiki.distributed.all_sum` and divide by the size of the
:class:`tiki.distributed.Group`. Namely we
have to :func:`tiki.utils.tree_map` the gradients with following function.

.. code:: python

    def all_avg(x):
        return tk.distributed.all_sum(x) / tk.distributed.init().size()

Putting everything together our training loop step looks as follows with
everything else remaining the same.

.. code:: python

    from tiki.utils import tree_map

    def all_reduce_grads(grads):
        N = tk.distributed.init().size()
        if N == 1:
            return grads
        return tree_map(
            lambda x: tk.distributed.all_sum(x) / N,
            grads
        )

    def step(model, x, y):
        loss, grads = loss_grad_fn(model, x, y)
        grads = all_reduce_grads(grads)  # <--- This line was added
        optimizer.update(model, grads)
        return loss

Using ``nn.average_gradients``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the code example above works correctly; it performs one communication
per gradient. It is significantly more efficient to aggregate several gradients
together and perform fewer communication steps.

This is the purpose of :func:`tiki.nn.average_gradients`. The final code looks
almost identical to the example above:

.. code:: python

    model = ...
    optimizer = ...
    dataset = ...

    def step(model, x, y):
        loss, grads = loss_grad_fn(model, x, y)
        grads = tk.nn.average_gradients(grads)  # <---- This line was added
        optimizer.update(model, grads)
        return loss

    for x, y in dataset:
        loss = step(model, x, y)
        tk.eval(loss, model.parameters())
