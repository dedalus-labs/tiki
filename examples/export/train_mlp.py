# Copyright © 2024 Apple Inc.

import tiki as tk
import tiki.nn as nn
import tiki.optimizers as optim
import tiki.utils


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)


if __name__ == "__main__":

    batch_size = 8
    input_dim = 32
    output_dim = 10

    def init():
        # Seed for the parameter initialization
        tk.random.seed(0)
        model = MLP(
            num_layers=3, input_dim=input_dim, hidden_dim=64, output_dim=output_dim
        )
        optimizer = optim.SGD(learning_rate=1e-1)
        optimizer.init(model.parameters())
        state = [model.parameters(), optimizer.state]
        tree_structure, state = zip(*tiki.utils.tree_flatten(state))
        return model, optimizer, tree_structure, state

    # Export the model parameter initialization
    model, optimizer, tree_structure, state = init()
    tk.eval(state)
    tk.export_function("init_mlp.tkfn", lambda: init()[-1])

    def loss_fn(params, X, y):
        model.update(params)
        return nn.losses.cross_entropy(model(X), y, reduction="mean")

    def step(*inputs):
        *state, X, y = inputs
        params, opt_state = tiki.utils.tree_unflatten(list(zip(tree_structure, state)))
        optimizer.state = opt_state
        loss, grads = tk.value_and_grad(loss_fn)(params, X, y)
        params = optimizer.apply_gradients(grads, params)
        _, state = zip(*tiki.utils.tree_flatten([params, optimizer.state]))
        return *state, loss

    # Make some random data
    tk.random.seed(42)
    example_X = tk.random.normal(shape=(batch_size, input_dim))
    example_y = tk.random.randint(low=0, high=output_dim, shape=(batch_size,))
    tk.export_function("train_mlp.tkfn", step, *state, example_X, example_y)

    # Export one step of SGD
    imported_step = tk.import_function("train_mlp.tkfn")

    for it in range(100):
        *state, loss = imported_step(*state, example_X, example_y)
        if it % 10 == 0:
            print(f"Loss {loss.item():.6}")
