#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "lidar_net/logging.h"
#include "lidar_net/lidar_net_det.h"
#include "lidar_net/object.h"
#include "common/json.hpp"
#include <string_view>

namespace lidar_net {

// impl类的基类
class LidarNetDetectorImpl {

public:
    virtual ~LidarNetDetectorImpl() = default;
    
    virtual bool Init(const std::string_view param_node_str, std::shared_ptr<LidarNetLogger> logger, const std::string& type) = 0;

    virtual std::vector<base::BoundingBox> Processing(const Eigen::MatrixXf& points, ProcessStatus* status, const bool& col_major = true) = 0;

    virtual void Processing(const Eigen::MatrixXf& points, std::vector<base::BoundingBox>& objects, ProcessStatus* status, const bool& col_major = true) = 0;

};

}  // namespace lidar_net
