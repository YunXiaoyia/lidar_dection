/********************************************************************************
 * @author: xx
 * @email: xx@hotmail.com
 * @date: 23-5-29 下午5:25
 * @version: 1.0
 * @description:
 ********************************************************************************/

#pragma once

#include "lidar_net/lidar_net_det.h"
#include "common/json.hpp"

namespace json {

bool ObjectsToJson(const std::vector<lidar_net::base::BoundingBox>& objects, nlohmann::json& json_object);

bool JsonToFile(const nlohmann::json& json_object, const std::string& file_name);

}  // namespace json
