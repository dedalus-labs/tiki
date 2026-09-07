template<int F, int N>
__device__ void tree_scan(float (&s)[F][N]) {
  for (int stride = 2; stride <= N; stride *= 2) {
    for (int node = threadIdx.x; node < N / stride; node += blockDim.x) {
      int right = (node + 1) * stride - 1, left = right - stride / 2;
      float ra = s[0][right];
      for (int field = 1; field < F; ++field)
        s[field][right] = ra * s[field][left] + s[field][right];
      s[0][right] = ra * s[0][left];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s[0][N - 1] = 1.f;
    for (int field = 1; field < F; ++field) s[field][N - 1] = 0.f;
  }
  __syncthreads();
  for (int stride = N; stride >= 2; stride /= 2) {
    for (int node = threadIdx.x; node < N / stride; node += blockDim.x) {
      int right = (node + 1) * stride - 1, left = right - stride / 2;
      float la = s[0][left], pa = s[0][right];
      for (int field = 1; field < F; ++field) {
        float lb = s[field][left], pb = s[field][right];
        s[field][left] = pb;
        // The first segment has an empty prefix, not an IEEE arithmetic identity.
        s[field][right] = node ? la * pb + lb : lb;
      }
      s[0][left] = pa;
      s[0][right] = la * pa;
    }
    __syncthreads();
  }
}
