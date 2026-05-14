/*
 * @Author: tianxiaosen Tream733@163.com
 * @Date: 2025-07-02 10:06:25
 * @LastEditors: tianxiaosen Tream733@163.com
 * @LastEditTime: 2025-07-02 10:38:31
 * @FilePath: /lidarnetsdk/src/blob/acl_buffer.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef acl_BUFFER_H
#define acl_BUFFER_H

#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <acl/ops/acl_dvpp.h>

#include <numeric>
#include <stdexcept>

#include "generic_buffer.h"

namespace lidar_net {
namespace infer {

/// @brief Functor for device aclrtMalloc
struct ACLDeviceAllocator {
    bool operator()(void** ptr, size_t size) const {
        return aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
};

/// @brief Functor for device free
struct ACLDeviceDestructor {
    void operator()(void* ptr) const { aclrtFree(ptr); }
};

/// @brief Functor for host paged memory aclrtMallocHost
struct ACLPinnedAllocator {
    bool operator()(void** ptr, size_t size) const { return aclrtMallocHost(ptr, size) == ACL_SUCCESS; }
};

/// @brief Functor for host paged memory free
struct ACLPinnedDestructor {
    void operator()(void* ptr) const { aclrtFreeHost(ptr); }
};

/// @brief Functor for dvpp acldvppMalloc
struct ACLDvppAllocator {
    bool operator()(void** ptr, size_t size) const { return acldvppMalloc(ptr, size) == ACL_SUCCESS; }
};

/// @brief Functor for dvpp
struct ACLDvppDestructor {
    void operator()(void* ptr) const { acldvppFree(ptr); }
};

/// @brief Buffer on common ACL device
using ACLDeviceBuffer = GenericBuffer<ACLDeviceAllocator, ACLDeviceDestructor>;

/// @brief Buffer on ACL pinned/paged memory
using ACLPinnedBuffer = GenericBuffer<ACLPinnedAllocator, ACLPinnedDestructor>;

/// @brief Buffer on ACL Dvpp memory
using ACLDvppBuffer = GenericBuffer<ACLDvppAllocator, ACLDvppDestructor>;

}  // namespace infer
}  // namespace lidar_net

#endif