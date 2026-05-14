#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#include <cuda_runtime_api.h>
#include <numeric>
#include <stdexcept>

#include "generic_buffer.h"

namespace lidar_net {
namespace infer {

/// @brief Functor for device cudaMalloc
struct CUDADeviceAllocator {
    bool operator()(void** ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
};

/// @brief Functor for device cudaFree
struct CUDADeviceDestructor {
    void operator()(void* ptr) const { cudaFree(ptr); }
};

/// @brief Functor for host/device cudaMallocManaged
struct CUDAUnifiedAllocator {
    bool operator()(void** ptr, size_t size) const { return cudaMallocManaged(ptr, size) == cudaSuccess; }
};

/// @brief Functor for pinned host memory cudaMallocHost
struct CUDAPinnedAllocator {
    bool operator()(void** ptr, size_t size) const { return cudaMallocHost(ptr, size) == cudaSuccess; }
};

/// @brief Functor for pinned host memory cudaFreeHost
struct CUDAPinnedDestructor {
    void operator()(void* ptr) const { cudaFreeHost(ptr); }
};

/// @brief Buffer on common cuda device
using CUDADeviceBuffer = GenericBuffer<CUDADeviceAllocator, CUDADeviceDestructor>;

/// @brief Buffer on cuda unified memory
using CUDAUnifiedBuffer = GenericBuffer<CUDAUnifiedAllocator, CUDADeviceDestructor>;

/// @brief Buffer on cuda pinned/paged memory
using CUDAPinnedBuffer = GenericBuffer<CUDAPinnedAllocator, CUDAPinnedDestructor>;

}  // namespace infer
}  // namespace lidar_net

#endif