
//
// Created by jie.gong on 22-02-17.
//

#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <map>

#include "lib/interface/base_target_track.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

enum class TrackMethod {
  SIMPLE_TRACK,
  L_SHAPE,
  UNKNOWN
};

static std::map<std::string, TrackMethod> track_method{
    {"simple_track", TrackMethod::SIMPLE_TRACK},
    {"l_shape", TrackMethod::L_SHAPE},
};

/**
 * @brief Target track method manager
 */
class TargetTrackManager {
 public:
  /**
   * @brief create a object tracking generate object
   * @param methodName optional: APOLLO_MLF, L_SHAPE
   * @return BaseTargetTrackerPtr generate pointer
   */
  static BaseTargetTrackPtr create(const TrackMethod& method);
};

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
