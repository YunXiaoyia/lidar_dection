#include "scatterMaxKernel.cuh"
#include <iostream>
#include <cuda_runtime_api.h>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS


__device__ __forceinline__ float atomicMax(float* address, float val) {
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret)) {
        int old = ret;
        if ((ret = atomicCAS((int*)address, old, __float_as_int(val))) == old) break;
    }
    return __int_as_float(ret);
}

__global__ void scatterMax_kernel(int max_num_points, int channels, int num_segments, int numel, const float* src_data,
                                  const int* index_info, float* out_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // int batch = thread_idx / (max_num_points * channels);
    int point_idx = (thread_idx / channels) % max_num_points;
    int channel_idx = thread_idx % channels;

    int pillar_idx = index_info[point_idx];
    if (thread_idx < numel && pillar_idx >= 0) {
        atomicMax(&out_data[pillar_idx * channels + channel_idx], src_data[thread_idx]);
    }
}


void scatterMaxKernelLaunch(cudaStream_t stream, int batch_size, int max_num_points, int channels, int num_segments,
                            const float* src_data, const int* index, float* out_data) {
    assert(batch_size == 1);
    int numel = batch_size * max_num_points * channels;  // number of elements
    int threads = THREADS;
    int blocks = BLOCKS(numel);
    scatterMax_kernel<<<blocks, threads, 0, stream>>>(max_num_points, channels, num_segments, numel, src_data, index,
                                                      out_data);
}
