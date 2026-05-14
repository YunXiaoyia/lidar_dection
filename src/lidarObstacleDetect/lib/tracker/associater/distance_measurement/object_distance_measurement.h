//
// Created by jie.gong on 22-03-17.
//
#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include "common/base.h"
#include "../../lidarnetsdk/src/common/toml.hpp"

#include "common/tracked_object.h"
#include "simple_track/tracklet.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

class ObjectDistanceMeasurement {
public:
    ObjectDistanceMeasurement();
    ~ObjectDistanceMeasurement() = default;

    bool Init();

    // @brief: compute object track distance
    // @params [in]: object
    // @params [in]: track data
    // @return: distance
    float ComputeDistance(const common::ObjectPtr& object,
                           const TrackletPtr& track);

private:

    std::vector<float> kDefaultWeight_;

    float bbox_iou_match_threshold_ = 0.8f;
};  // class ObjectDistanceMeasurement

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

