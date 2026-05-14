
//
// Created by jie.gong on 22-02-17.
//

#pragma once

#include "common/tracked_object.h"
#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

class MultiAnchorMotionMeasurement {
 public:
  MultiAnchorMotionMeasurement() = default;
  ~MultiAnchorMotionMeasurement() = default;
  // @brief: wrapper of motion measurement functions
  // @params [in]: track_data: track data
  // @params [in]: new_object: object for current updating
  void ComputeMotionMeasurment(const TrackedObjectConstPtr& track_data,
                               TrackedObjectPtr& new_object);

 protected:
  // @brief: select measurement based on track history
  // @params [in]: track data
  // @params [in]: latest object in track
  // @params [out]: new object for storing selection
  void MeasurementSelection(const TrackedObjectConstPtr& latest_object,
                            TrackedObjectPtr& new_object);

  void MeasurementSelection_from_cui(const TrackedObjectConstPtr& latest_object,
                            TrackedObjectPtr& new_object, const double& time_diff);
  // @brief: estimate measurement quality
  // @params [in]: latest object in track
  // @params [out]: new object for storing quality
  void MeasurementQualityEstimation(const TrackedObjectConstPtr& latest_object,
                                    TrackedObjectPtr& new_object);

 private:
  const double EPSILON_TIME = 1e-3;  // or numeric_limits<double>::epsilon()
  const double DEFAULT_FPS = 0.1;
};  // class MultiAnchorMotionMeasurement

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

