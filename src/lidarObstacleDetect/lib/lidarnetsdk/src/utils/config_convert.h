#pragma once

#include <string_view>

namespace lidar_net {

std::string convert2json(std::string_view config_path, std::string_view config_type);
std::string config_convert(std::string_view param_node_str, std::string_view config_type);

}  // namespace lidar_net