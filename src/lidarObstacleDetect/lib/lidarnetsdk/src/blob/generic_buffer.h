#ifndef COMMON_BLOB_GENERIC_BUFFER
#define COMMON_BLOB_GENERIC_BUFFER

#include <cstdlib>
#include <numeric>
#include <stdexcept>

#include "infer_common.h"

#ifdef USING_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef USING_ACL
#include <acl/acl_base.h>
#include <acl/ops/acl_dvpp.h>
#endif

namespace lidar_net {
namespace infer {

/**
 * @brief 内存分配、释放的RAII模板类
 *
 * @tparam AllocFunc 分配内存的仿函数
 * @tparam FreeFunc 释放内存的仿函数
 */
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
   public:
    /// @brief Construct an empty buffer.
    GenericBuffer(DataType type = DataType::kFLOAT) : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr) {}

    /// @brief Construct a buffer with the specified allocation size in bytes.
    GenericBuffer(size_t size, DataType type) : mSize(size), mCapacity(size), mType(type) {
        if (!allocFn(reinterpret_cast<void**>(&mBuffer), this->nbytes())) {
            throw std::bad_alloc();
        }
    }

    /// @brief Move construction
    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType), mBuffer(buf.mBuffer) {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = DataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    /// @brief Move assignment
    GenericBuffer& operator=(GenericBuffer&& buf) {
        if (this != &buf) {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    /// @brief Returns pointer to underlying array.
    void* data() { return mBuffer; }

    /// @brief Returns pointer to underlying array.
    const void* data() const { return mBuffer; }

    /// @brief Returns the size (in number of elements) of the buffer.
    size_t size() const { return mSize; }

    /// @brief Returns the capacity (in number of elements) of the buffer.
    size_t capacity() const { return mCapacity; }

    /// @brief Returns the size (in bytes) of the buffer.
    size_t nbytes() const { return this->size() * kDataTypeSizeMap.at(mType); }

    /// @brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    void resize(size_t newSize) {
        mSize = newSize;
        if (mCapacity < newSize) {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbytes())) {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }

    /// @brief Overload of resize that accepts Dims
    void resize(const Dims& dims) {
        // return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
        return this->resize(dims.numel());
    }

    /// @brief Destruction: free memory
    ~GenericBuffer() { freeFn(mBuffer); }

   private:
    size_t mSize, mCapacity;
    DataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

/// @brief Functor for host malloc
struct HostAllocator {
    bool operator()(void** ptr, size_t size) const {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

/// @brief Functor for host free
struct HostDestructor {
    void operator()(void* ptr) const { free(ptr); }
};

/// @brief Buffer for generic buffer for common c/cpp
using HostBuffer = GenericBuffer<HostAllocator, HostDestructor>;

}  // namespace infer
}  // namespace lidar_net

#endif /* COMMON_BLOB_GENERIC_BUFFER */
