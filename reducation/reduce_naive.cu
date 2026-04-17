#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

#define CHECK_CUDA(call) \
do { \
    cudaError_t err=(call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void reduce(const float *input, float *output, int n) {
    __shared__ float idata[BLOCK_SIZE]; 
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    idata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * tid;
        if (index + i < blockDim.x) {
            idata[index] += idata[index + i]; 
        }
        __syncthreads();
    }
    if (tid == 0) { 
        atomicAdd(output, idata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int grid_size = (N - 1) / BLOCK_SIZE + 1;
    reduce<<<grid_size, BLOCK_SIZE>>>(input, output, N);
}

int main() {
    const int N = 8;
    float h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_output = 0.0f;

    float *d_input = NULL;
    float *d_output = NULL;

    CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_input, d_output, N);

    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    printf("result = %f\n", h_output);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    return 0;
}