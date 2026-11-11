#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int kN) {
    int idx = blockIdx.x  * blockDim.x + threadIdx.x;
    if(idx > N - kN +1) return;
    float tmp = 0.0f;
    for(int k=0; k < kN; k++){
        tmp += input[k + idx] * kernel[k];
    }
    output[idx] = tmp;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threads = 256;
    dim3 blocks((outN + 255) / 256);
    conv1d_kernel<<<blocks, threads>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}
