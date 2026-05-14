#pragma once
#include <cmath>

namespace depthink {
namespace perception {
namespace lidar {
namespace utils {

inline bool inFOV(float x, float y, float fov_deg, float dist_threshold) {
    // 简单的 FOV 判断逻辑
    // x 为前向
    if (x < 0) return false; 
    float angle = std::atan2(std::abs(y), x) * 180.0f / M_PI;
    return angle <= (fov_deg / 2.0f);
}

}
}
}
}