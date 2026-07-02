extern "C" __global__ void add(float *x, float *y, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = x[idx] + y[idx];
}
