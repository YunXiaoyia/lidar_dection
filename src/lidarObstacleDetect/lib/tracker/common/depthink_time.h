#pragma once

#include <chrono>
#include <string>

#include "common/Macros.h"

HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {
namespace common {

// Simple time utility class
class DepthInkTime {
public:
    static double GetCurrentTime() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration<double>(duration).count();
    }
    
    static std::string GetCurrentTimeString() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        return std::to_string(time_t);
    }
};

} // namespace common
} // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
