
#pragma once

#include "common/tracked_object.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

// @brief: compute location distance for given track & object
// @params [in]: object for computing distance
// @params [in]: predicted state of track for computing distance
// @params [in]: new detected object for computing distance
// @params [in]: time interval from last matching
// @return: distance
float LocationDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff);

// @brief: compute direction distance for given track & object
// @params [in]: object for computing distance
// @params [in]: predicted state of track for computing distance
// @params [in]: new detected object for computing distance
// @params [in]: time interval from last matching
// @return distance
float DirectionDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff);

// @brief: compute bbox size distance for given track & object
// @params [in]: object for computing distance
// @params [in]: predicted state of track for computing distance
// @params [in]: new detected object for computing distance
// @params [in]: time interval from last matching
// @return distance
float BboxSizeDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff);

// @brief: compute point num distance for given track & object
// @params [in]: object for computing distance
// @params [in]: predicted state of track for computing distance
// @params [in]: new detected object for computing distance
// @params [in]: time interval from last matching
// @return distance
float PointNumDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff);

// @brief: compute histogram distance for given track & object
// @params [in]: object for computing distance
// @params [in]: predicted state of track for computing distance
// @params [in]: new detected object for computing distance
// @params [in]: time interval from last matching
// @return distance
float HistogramDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff);

// @brief: compute centroid shift distance for object and background match
// @params [in]: object for computing distance
// @params [in]: unused
// @params [in]: new detected object for computing distance
// @params [in]: unused
// @return distance
float CentroidShiftDistance(const TrackedObjectConstPtr& last_object,
                            const common::ObjectPtr& object,
                            const double time_diff);

// @brief compute bbox iou distance for object and background match
// @params [in]: object for computing distance
// @params [in]: unused
// @params [in]: new detected object for computing distance
// @params [in]: unused
// @return distance
float BboxIouDistance(const TrackedObjectConstPtr& last_object,
                      const common::ObjectPtr& object,
                      const double time_diff, float match_threshold);

// @brief lidar only: compute semantic map based distance
// @params [in]: track data contained predicted trajectory feature
// @params [in]: new detected object for computing distance
// @return distance
// float SemanticMapDistance(const MlfTrackData& track_dat,
//                          const TrackedObjectConstPtr& cur_obj);


// @brief: compute box tail distance for given track & object
// @params [in]: object for computing distance
// @params [in]: predicted state of track for computing distance
// @params [in]: new detected object for computing distance
// @params [in]: time interval from last matching
// @return: distance
float BboxTailDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff);

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

