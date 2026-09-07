# Copyright © 2024 Apple Inc.

import matplotlib
import numpy as np
import sympy
import tiki as tk
import torch
from time_utils import measure_runtime

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bandwidth_gb(runtime_ms, system_size):
    bytes_per_fft = np.dtype(np.complex64).itemsize * 2
    bytes_per_gb = 1e9
    ms_per_s = 1e3
    return system_size * bytes_per_fft / runtime_ms * ms_per_s / bytes_per_gb


def run_bench(system_size, fft_sizes, backend="tiki", dim=1):
    def fft_tiki(x):
        if dim == 1:
            out = tk.fft.fft(x)
        elif dim == 2:
            out = tk.fft.fft2(x)
        tk.eval(out)
        return out

    def fft_mps(x):
        if dim == 1:
            out = torch.fft.fft(x)
        elif dim == 2:
            out = torch.fft.fft2(x)
        torch.mps.synchronize()
        return out

    bandwidths = []
    for n in fft_sizes:
        batch_size = system_size // n**dim
        shape = [batch_size] + [n for _ in range(dim)]
        if backend == "tiki":
            x_np = np.random.uniform(size=(system_size // n, n)).astype(np.complex64)
            x = tk.array(x_np)
            tk.eval(x)
            fft = fft_tiki
        elif backend == "mps":
            x_np = np.random.uniform(size=(system_size // n, n)).astype(np.complex64)
            x = torch.tensor(x_np, device="mps")
            torch.mps.synchronize()
            fft = fft_mps
        else:
            raise NotImplementedError()
        runtime_ms = measure_runtime(fft, x=x)
        bandwidth = bandwidth_gb(runtime_ms, np.prod(shape))
        print(n, bandwidth)
        bandwidths.append(bandwidth)

    return np.array(bandwidths)


def time_fft():
    x = np.array(range(2, 512))
    system_size = int(2**26)

    print("Tiki GPU")
    with tk.stream(tk.gpu):
        gpu_bandwidths = run_bench(system_size=system_size, fft_sizes=x)

    print("MPS GPU")
    mps_bandwidths = run_bench(system_size=system_size, fft_sizes=x, backend="mps")

    print("CPU")
    system_size = int(2**20)
    with tk.stream(tk.cpu):
        cpu_bandwidths = run_bench(system_size=system_size, fft_sizes=x)

    x = np.array(x)

    all_indices = x - x[0]
    radix_2to13 = (
        np.array([i for i in x if all(p <= 13 for p in sympy.primefactors(i))]) - x[0]
    )
    bluesteins = (
        np.array([i for i in x if any(p > 13 for p in sympy.primefactors(i))]) - x[0]
    )

    for indices, name in [
        (all_indices, "All"),
        (radix_2to13, "Radix 2-13"),
        (bluesteins, "Bluestein's"),
    ]:
        # plot bandwidths
        print(name)
        plt.scatter(x[indices], gpu_bandwidths[indices], color="green", label="GPU")
        plt.scatter(x[indices], mps_bandwidths[indices], color="blue", label="MPS")
        plt.scatter(x[indices], cpu_bandwidths[indices], color="red", label="CPU")
        plt.title(f"Tiki FFT Benchmark -- {name}")
        plt.xlabel("N")
        plt.ylabel("Bandwidth (GB/s)")
        plt.legend()
        plt.savefig(f"{name}.png")
        plt.clf()

    av_gpu_bandwidth = np.mean(gpu_bandwidths)
    av_mps_bandwidth = np.mean(mps_bandwidths)
    av_cpu_bandwidth = np.mean(cpu_bandwidths)
    print("Average bandwidths:")
    print("GPU:", av_gpu_bandwidth)
    print("MPS:", av_mps_bandwidth)
    print("CPU:", av_cpu_bandwidth)

    portion_faster = len(np.where(gpu_bandwidths > mps_bandwidths)[0]) / len(x)
    print("Percent Tiki faster than MPS: ", portion_faster * 100)


if __name__ == "__main__":
    time_fft()
