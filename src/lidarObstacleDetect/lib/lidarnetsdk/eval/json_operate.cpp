/********************************************************************************
 * @author: xxx
 * @email: xxx@hotmail.com
 * @date: 23-5-29 下午5:25
 * @version: 1.0
 * @description:
 ********************************************************************************/
#include "json_operate.h"

#include <fstream>
#include <iostream>

namespace json {

bool ObjectsToJson(const std::vector<lidar_net::base::BoundingBox>& objects, nlohmann::json& json_object) {
  int count = 0;
  nlohmann::json lidar_detected_objs;
  for (const auto& obj : objects) {
    nlohmann::json j_obj;
    j_obj["obj_id"] = std::to_string(count);
    j_obj["obj_type"] = lidar_net::base::ObjectEnumStrDict.at(obj.type);
    j_obj["confidence"] = obj.confidence;
    std::vector<float> type_probs(obj.type_probs.data(), obj.type_probs.data() + obj.type_probs.size());
    j_obj["type_probs"] = type_probs;
    j_obj["visibility"] = "unknown";
    j_obj["psr"]["position"]["x"] = obj.center.x();
    j_obj["psr"]["position"]["y"] = obj.center.y();
    j_obj["psr"]["position"]["z"] = obj.center.z();
    j_obj["psr"]["scale"]["x"] = obj.size.x();
    j_obj["psr"]["scale"]["y"] = obj.size.y();
    j_obj["psr"]["scale"]["z"] = obj.size.z();
    j_obj["psr"]["rotation"]["x"] = 0.0;
    j_obj["psr"]["rotation"]["y"] = 0.0;
    j_obj["psr"]["rotation"]["z"] = obj.theta;
    for (int i = 0; i < 8; ++i) {
      nlohmann::json points;
      //            points.push_back(obj.corners2d(i % 4, 0));
      //            points.push_back(obj.corners2d(i % 4, 1));
      points.push_back(0.0);
      points.push_back(0.0);
      if (i > 3) {
        points.push_back(2.0f);
      } else {
        points.push_back(0.0f);
      }
      points.push_back(1);
      j_obj["vertices"].push_back(points);
    }
    // j_obj["points_in_bbox"] = 0; // invalid

    lidar_detected_objs.push_back(j_obj);
  }
  json_object["lidar"]["detected_objs"] = lidar_detected_objs;

  return true;
}

bool JsonToFile(const nlohmann::json& json_object, const std::string& file_name) {
  std::ofstream outfile(file_name);
  std::string serialized_string = json_object.dump(4);
  if (outfile.is_open()) {
    outfile << serialized_string << std::endl;
    outfile.close();
  }
  return true;
}

}  // namespace json
