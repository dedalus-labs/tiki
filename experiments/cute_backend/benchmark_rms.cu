// Matched scalar RMSNorm schedule for comparing NVCC and CuTe code generation.
#ifndef ROW_THREADS
#define ROW_THREADS 32
#endif
#ifndef BLOCK_ROWS
#define BLOCK_ROWS 4
#endif

extern "C" __global__ void
rms_reference(const float* x, const float* weight, float* output) {
  const int row = blockIdx.x * BLOCK_ROWS + threadIdx.x / ROW_THREADS;
  const int column_start = threadIdx.x % ROW_THREADS;
  const int lane = threadIdx.x % 32;
  constexpr int warps = ROW_THREADS / 32;
  __shared__ float scratch[BLOCK_ROWS * (warps > 1 ? warps : 1)];
  float total = 0.0f;
  for (int column = column_start; column < WIDTH; column += ROW_THREADS) {
    if (row < ROWS) {
      const float value = x[row * WIDTH + column];
      total += value * value;
    }
  }
  for (int offset = (ROW_THREADS < 32 ? ROW_THREADS : 32) / 2; offset;
       offset /= 2) {
    total += __shfl_xor_sync(0xffffffff, total, offset);
  }
  if constexpr (warps > 1) {
    const int local_row = threadIdx.x / ROW_THREADS;
    if (lane == 0) {
      scratch[local_row * warps + column_start / 32] = total;
    }
    __syncthreads();
    total = lane < warps ? scratch[local_row * warps + lane] : 0.0f;
    for (int offset = 16; offset; offset /= 2) {
      total += __shfl_xor_sync(0xffffffff, total, offset);
    }
  }
  const float inv_rms = rsqrtf(total * (1.0f / WIDTH) + 1e-6f);
  if (row < ROWS) {
    for (int column = column_start; column < WIDTH; column += ROW_THREADS) {
      output[row * WIDTH + column] =
          x[row * WIDTH + column] * inv_rms * weight[column];
    }
  }
}
