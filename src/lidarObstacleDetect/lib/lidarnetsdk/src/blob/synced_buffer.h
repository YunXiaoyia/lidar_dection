#ifndef COMMON_BLOB_SYNCED_BUFFER
#define COMMON_BLOB_SYNCED_BUFFER

#include <iostream>
#include <numeric>
#include <stdexcept>

#include "generic_buffer.h"
#include "infer_common.h"

#ifdef USING_CUDA
#include <cuda_runtime_api.h>

#include "cuda_buffer.h"
#endif

#ifdef USING_ACL
#include <acl_rt.h>
#include <dvpp/Vpc.h>
#include <dvpp/dvpp_config.h>
#include <dvpp/idvppapi.h>

#include "acl_buffer.h"
#endif

namespace lidar_net {
namespace infer {

/**
 * @brief Abstract Base Class for all __SyncedBuffer__ classes
 */
template <typename T>
class __SyncedBufferABC__ {
   public:
    __SyncedBufferABC__() = default;
    virtual ~__SyncedBufferABC__() = default;
    virtual inline void memcpyBuffers(const bool deviceToHost, const bool async, const T& stream = 0) = 0;
    virtual inline void* getDeviceBuffer() = 0;
    virtual inline const void* getDeviceBuffer() const = 0;
    virtual inline void* getHostBuffer() = 0;
    virtual inline const void* getHostBuffer() const = 0;
    virtual inline void* get() = 0;
    virtual inline const void* get() const = 0;
    virtual inline size_t nbytes() const = 0;
    virtual inline void resize(size_t newSize) = 0;
    virtual inline size_t size() const = 0;
    virtual inline size_t capacity() const = 0;
};

#ifdef USING_CUDA

/**
 * @brief Global memory allocation
 * Device: cudaMalloc; Host: malloc
 */
class CudaSyncedBufferGlobal : public __SyncedBufferABC__<cudaStream_t> {
   public:
    CudaSyncedBufferGlobal(size_t vol, DataType type) : deviceBuffer(vol, type), hostBuffer(vol, type) {}
    inline void memcpyBuffers(const bool deviceToHost, const bool async, const cudaStream_t& stream = 0) final {
        void* dstPtr = deviceToHost ? hostBuffer.data() : deviceBuffer.data();
        const void* srcPtr = deviceToHost ? deviceBuffer.data() : hostBuffer.data();
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        const size_t byteSize = hostBuffer.nbytes();
        if (async)
            cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream);
        else
            cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);
    }
    inline void* getDeviceBuffer() final { return deviceBuffer.data(); }
    inline const void* getDeviceBuffer() const final { return deviceBuffer.data(); }
    inline void* getHostBuffer() final { return hostBuffer.data(); }
    inline const void* getHostBuffer() const final { return hostBuffer.data(); }
    inline void* get() final { return deviceBuffer.data(); }
    inline const void* get() const final { return deviceBuffer.data(); }
    inline size_t nbytes() const final { return deviceBuffer.nbytes(); }
    inline void resize(size_t newSize) final {
        deviceBuffer.resize(newSize);
        hostBuffer.resize(newSize);
    }
    inline size_t size() const final { return deviceBuffer.size(); }
    inline size_t capacity() const final { return deviceBuffer.capacity(); }

   private:
    CUDADeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

/**
 * @brief Unified memory allocation
 * Device & Host: cudaMallocManaged;
 */
class CudaSyncedBufferUnified : public __SyncedBufferABC__<cudaStream_t> {
   public:
    CudaSyncedBufferUnified(size_t vol, DataType type) : unifiedBuffer(vol, type) {}
    inline void memcpyBuffers(const bool deviceToHost, const bool async, const cudaStream_t& stream = 0) final {}
    inline void* getDeviceBuffer() final { return unifiedBuffer.data(); }
    inline const void* getDeviceBuffer() const final { return unifiedBuffer.data(); }
    inline void* getHostBuffer() final { return unifiedBuffer.data(); }
    inline const void* getHostBuffer() const final { return unifiedBuffer.data(); }
    inline void* get() final { return unifiedBuffer.data(); }
    inline const void* get() const final { return unifiedBuffer.data(); }
    inline size_t nbytes() const final { return unifiedBuffer.nbytes(); }
    inline void resize(size_t newSize) final { unifiedBuffer.resize(newSize); }
    inline size_t size() const final { return unifiedBuffer.size(); }
    inline size_t capacity() const final { return unifiedBuffer.capacity(); }

   private:
    CUDAUnifiedBuffer unifiedBuffer;
};

/**
 * @brief Pinned memory allocation
 * Device: cudaMalloc; Host: cudaMallocHost
 */
class CudaSyncedBufferPinned : public __SyncedBufferABC__<cudaStream_t> {
   public:
    CudaSyncedBufferPinned(size_t vol, DataType type) : deviceBuffer(vol, type), pinnedBuffer(vol, type) {}
    inline void memcpyBuffers(const bool deviceToHost, const bool async, const cudaStream_t& stream = 0) final {
        void* dstPtr = deviceToHost ? pinnedBuffer.data() : deviceBuffer.data();
        const void* srcPtr = deviceToHost ? deviceBuffer.data() : pinnedBuffer.data();
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        const size_t byteSize = pinnedBuffer.nbytes();
        if (async)
            cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream);
        else
            cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);
    }
    inline void* getDeviceBuffer() final { return deviceBuffer.data(); }
    inline const void* getDeviceBuffer() const final { return deviceBuffer.data(); }
    inline void* getHostBuffer() final { return pinnedBuffer.data(); }
    inline const void* getHostBuffer() const final { return pinnedBuffer.data(); }
    inline void* get() final { return deviceBuffer.data(); }
    inline const void* get() const final { return deviceBuffer.data(); }
    inline size_t nbytes() const final { return deviceBuffer.nbytes(); }
    inline void resize(size_t newSize) final {
        deviceBuffer.resize(newSize);
        pinnedBuffer.resize(newSize);
    }
    inline size_t size() const final { return deviceBuffer.size(); }
    inline size_t capacity() const final { return deviceBuffer.capacity(); }

   private:
    CUDADeviceBuffer deviceBuffer;
    CUDAPinnedBuffer pinnedBuffer;
};

using SyncedBuffer = CudaSyncedBufferPinned;

#endif

#ifdef USING_ACL

/**
 * @brief Global memory allocation
 * Device: cudaMalloc; Host: malloc
 */
class AclSyncedBufferGlobal : public __SyncedBufferABC__<aclrtStream> {
   public:
    AclSyncedBufferGlobal(size_t vol, DataType type) : deviceBuffer(vol, type), hostBuffer(vol, type) {}
    inline void memcpyBuffers(const bool deviceToHost, const bool async, const aclrtStream& stream = 0) final {
        void* dstPtr = deviceToHost ? hostBuffer.data() : deviceBuffer.data();
        const void* srcPtr = deviceToHost ? deviceBuffer.data() : hostBuffer.data();
        const aclrtMemcpyKind memcpyType = deviceToHost ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;
        const size_t byteSize = hostBuffer.nbytes();
        if (async)
            aclrtMemcpyAsync(dstPtr, byteSize, srcPtr, byteSize, memcpyType, stream);
        else
            aclrtMemcpy(dstPtr, byteSize, srcPtr, byteSize, memcpyType);
    }
    inline void* getDeviceBuffer() final { return deviceBuffer.data(); }
    inline const void* getDeviceBuffer() const final { return deviceBuffer.data(); }
    inline void* getHostBuffer() final { return hostBuffer.data(); }
    inline const void* getHostBuffer() const final { return hostBuffer.data(); }
    inline void* get() final { return deviceBuffer.data(); }
    inline const void* get() const final { return deviceBuffer.data(); }
    inline size_t nbytes() const final { return deviceBuffer.nbytes(); }
    inline void resize(size_t newSize) final {
        deviceBuffer.resize(newSize);
        hostBuffer.resize(newSize);
    }
    inline size_t size() const final { return deviceBuffer.size(); }
    inline size_t capacity() const final { return deviceBuffer.capacity(); }

   private:
    ACLDeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

/**
 * @brief Pinned memory allocation
 * Device: aclrtMalloc; Host: aclrtMallocHost
 */
class AclSyncedBufferPinned : public __SyncedBufferABC__<aclrtStream> {
   public:
    AclSyncedBufferPinned(size_t vol, DataType type) : deviceBuffer(vol, type), hostBuffer(vol, type) {}
    inline void memcpyBuffers(const bool deviceToHost, const bool async, const aclrtStream& stream = 0) final {
        void* dstPtr = deviceToHost ? hostBuffer.data() : deviceBuffer.data();
        const void* srcPtr = deviceToHost ? deviceBuffer.data() : hostBuffer.data();
        const aclrtMemcpyKind memcpyType = deviceToHost ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;
        const size_t byteSize = hostBuffer.nbytes();
        if (async)
            aclrtMemcpyAsync(dstPtr, byteSize, srcPtr, byteSize, memcpyType, stream);
        else
            aclrtMemcpy(dstPtr, byteSize, srcPtr, byteSize, memcpyType);
    }
    inline void* getDeviceBuffer() final { return deviceBuffer.data(); }
    inline const void* getDeviceBuffer() const final { return deviceBuffer.data(); }
    inline void* getHostBuffer() final { return hostBuffer.data(); }
    inline const void* getHostBuffer() const final { return hostBuffer.data(); }
    inline void* get() final { return deviceBuffer.data(); }
    inline const void* get() const final { return deviceBuffer.data(); }
    inline size_t nbytes() const final { return deviceBuffer.nbytes(); }
    inline void resize(size_t newSize) final {
        deviceBuffer.resize(newSize);
        hostBuffer.resize(newSize);
    }
    inline size_t size() const final { return deviceBuffer.size(); }
    inline size_t capacity() const final { return deviceBuffer.capacity(); }

   private:
    ACLDeviceBuffer deviceBuffer;
    ACLPinnedBuffer hostBuffer;
};

/**
 * @brief Device memory allocation
 * Device & Host: cudaMallocManaged;
 */
class AclSyncedBufferDevice : public __SyncedBufferABC__<aclrtStream> {
   public:
    AclSyncedBufferDevice(size_t vol, DataType type) : deviceBuffer(vol, type) {}
    inline void memcpyBuffers(const bool deviceToHost, const bool async, const aclrtStream& stream = 0) final {}
    inline void* getDeviceBuffer() final { return deviceBuffer.data(); }
    inline const void* getDeviceBuffer() const final { return deviceBuffer.data(); }
    inline void* getHostBuffer() final { return deviceBuffer.data(); }
    inline const void* getHostBuffer() const final { return deviceBuffer.data(); }
    inline void* get() final { return deviceBuffer.data(); }
    inline const void* get() const final { return deviceBuffer.data(); }
    inline size_t nbytes() const final { return deviceBuffer.nbytes(); }
    inline void resize(size_t newSize) final { deviceBuffer.resize(newSize); }
    inline size_t size() const final { return deviceBuffer.size(); }
    inline size_t capacity() const final { return deviceBuffer.capacity(); }

   private:
    ACLDeviceBuffer deviceBuffer;
};

/**
 * @brief Pinned memory allocation
 * Device: aclrtMalloc; Host: aclrtMallocHost
 */
class AclSyncedBufferDvpp : public __SyncedBufferABC__<aclrtStream> {
   public:
    AclSyncedBufferDvpp(size_t vol, DataType type) : deviceBuffer(vol, type), hostBuffer(vol, type) {}
    inline void memcpyBuffers(const bool deviceToHost, const bool async, const aclrtStream& stream = 0) final {
        void* dstPtr = deviceToHost ? hostBuffer.data() : deviceBuffer.data();
        const void* srcPtr = deviceToHost ? deviceBuffer.data() : hostBuffer.data();
        const aclrtMemcpyKind memcpyType = deviceToHost ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;
        const size_t byteSize = hostBuffer.nbytes();
        if (async)
            aclrtMemcpyAsync(dstPtr, byteSize, srcPtr, byteSize, memcpyType, stream);
        else
            aclrtMemcpy(dstPtr, byteSize, srcPtr, byteSize, memcpyType);
    }
    inline void* getDeviceBuffer() final { return deviceBuffer.data(); }
    inline const void* getDeviceBuffer() const final { return deviceBuffer.data(); }
    inline void* getHostBuffer() final { return hostBuffer.data(); }
    inline const void* getHostBuffer() const final { return hostBuffer.data(); }
    inline void* get() final { return deviceBuffer.data(); }
    inline const void* get() const final { return deviceBuffer.data(); }
    inline size_t nbytes() const final { return deviceBuffer.nbytes(); }
    inline void resize(size_t newSize) final {
        deviceBuffer.resize(newSize);
        hostBuffer.resize(newSize);
    }
    inline size_t size() const final { return deviceBuffer.size(); }
    inline size_t capacity() const final { return deviceBuffer.capacity(); }

   private:
    ACLDvppBuffer deviceBuffer;
    ACLPinnedBuffer hostBuffer;
};

using SyncedBuffer = AclSyncedBufferPinned;

#endif

struct SyncedTensor {
    SyncedTensor(BlobInfo info) : info(info) {
        auto vol = info.numel();
        buffer = std::make_unique<SyncedBuffer>(vol, info.dtype);
    }

    inline void resize(const Dims& shape) {
        info.shape = shape;
        auto vol = info.numel();
        buffer->resize(vol);
    }

    inline size_t numel() const { return info.numel(); }

    inline size_t nbytes() const { return info.nbytes(); }

    std::unique_ptr<SyncedBuffer> buffer;
    BlobInfo info;
};

}  // namespace infer
}  // namespace lidar_net

#endif /* COMMON_BLOB_SYNCED_BUFFER */
