#pragma once

#include "json.hpp"
#include "lidar_net/logging.h"
#include "log/logger.h"

namespace lidar_net {
template <typename T>
inline bool get_value(const nlohmann::json& t, const std::string& key, T& value) noexcept {
    if (t.contains(key)) {
        try {
            value = t[key].get<T>();
            return true;
        } catch (...) {
            PLOG_ERROR << "[Detector] Invalid config file: " << key << " failed to parse correctly";
            return false;
        }
    }
    PLOG_ERROR << "[Detector] Invalid config file: missing key " << key;
    return false;
}

#define GET_VALUE(KEY, VAL) \
    if (!get_value(KEY, #VAL, VAL)) return false;

}  // namespace lidar_net
