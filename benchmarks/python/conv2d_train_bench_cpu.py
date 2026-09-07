import time

import tiki as tk
import tiki.nn
import tiki.optimizers as opt
import torch


def bench_tiki(steps: int = 20) -> float:
    tk.set_default_device(tk.cpu)

    class BenchNetTiki(tiki.nn.Module):
        # simple encoder-decoder net

        def __init__(self, in_channels, hidden_channels=32):
            super().__init__()

            self.net = tiki.nn.Sequential(
                tiki.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                tiki.nn.ReLU(),
                tiki.nn.Conv2d(
                    hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1
                ),
                tiki.nn.ReLU(),
                tiki.nn.ConvTranspose2d(
                    2 * hidden_channels, hidden_channels, kernel_size=3, padding=1
                ),
                tiki.nn.ReLU(),
                tiki.nn.ConvTranspose2d(
                    hidden_channels, in_channels, kernel_size=3, padding=1
                ),
            )

        def __call__(self, input):
            return self.net(input)

    benchNet = BenchNetTiki(3)
    tk.eval(benchNet.parameters())
    optim = opt.Adam(learning_rate=1e-3)

    inputs = tk.random.normal([10, 256, 256, 3])

    params = benchNet.parameters()
    optim.init(params)

    state = [benchNet.state, optim.state]

    def loss_fn(params, image):
        benchNet.update(params)
        pred_image = benchNet(image)
        return (pred_image - image).abs().mean()

    def step(params, image):
        loss, grads = tk.value_and_grad(loss_fn)(params, image)
        optim.update(benchNet, grads)
        return loss

    total_time = 0.0
    print("Tiki:")
    for i in range(steps):
        start_time = time.perf_counter()

        step(benchNet.parameters(), inputs)
        tk.eval(state)
        end_time = time.perf_counter()

        print(f"{i:3d}, time={(end_time-start_time) * 1000:7.2f} ms")
        total_time += (end_time - start_time) * 1000

    return total_time


def bench_torch(steps: int = 20) -> float:
    device = torch.device("cpu")

    class BenchNetTorch(torch.nn.Module):
        # simple encoder-decoder net

        def __init__(self, in_channels, hidden_channels=32):
            super().__init__()

            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1
                ),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(
                    2 * hidden_channels, hidden_channels, kernel_size=3, padding=1
                ),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(
                    hidden_channels, in_channels, kernel_size=3, padding=1
                ),
            )

        def forward(self, input):
            return self.net(input)

    benchNet = BenchNetTorch(3).to(device)
    optim = torch.optim.Adam(benchNet.parameters(), lr=1e-3)

    inputs = torch.randn(10, 3, 256, 256, device=device)

    def loss_fn(pred_image, image):
        return (pred_image - image).abs().mean()

    total_time = 0.0
    print("PyTorch:")
    for i in range(steps):
        start_time = time.perf_counter()

        optim.zero_grad()
        pred_image = benchNet(inputs)
        loss = loss_fn(pred_image, inputs)
        loss.backward()
        optim.step()

        end_time = time.perf_counter()

        print(f"{i:3d}, time={(end_time-start_time) * 1000:7.2f} ms")
        total_time += (end_time - start_time) * 1000

    return total_time


def main():
    steps = 20
    time_tiki = bench_tiki(steps)
    time_torch = bench_torch(steps)

    print(f"average time of Tiki:     {time_tiki/steps:9.2f} ms")
    print(f"total time of Tiki:       {time_tiki:9.2f} ms")
    print(f"average time of PyTorch: {time_torch/steps:9.2f} ms")
    print(f"total time of PyTorch:   {time_torch:9.2f} ms")

    diff = time_torch / time_tiki - 1.0
    print(f"torch/tiki diff: {100. * diff:+5.2f}%")


if __name__ == "__main__":
    main()
