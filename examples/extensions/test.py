import tiki as tk
from tiki_sample_extensions import axpby

a = tk.ones((3, 4))
b = tk.ones((3, 4))
c_cpu = axpby(a, b, 4.0, 2.0, stream=tk.cpu)
c_gpu = axpby(a, b, 4.0, 2.0, stream=tk.gpu)

print(f"c shape: {c_cpu.shape}")
print(f"c dtype: {c_cpu.dtype}")
print(f"c_cpu correct: {tk.all(c_cpu == 6.0).item()}")
print(f"c_gpu correct: {tk.all(c_gpu == 6.0).item()}")
