
#pragma once

#include "common/tracked_object.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

// @brief: measure anchor point velocity
// @params [in/out]: new object for current updating
// @params [in]: old object for last updating
// @params [in]: time interval from last updating
void MeasureAnchorPointVelocity(TrackedObjectPtr new_object,
                                const TrackedObjectConstPtr& old_object,
                                const double& time_diff);


// @brief: measure detect center point velocity
// @params [in/out]: new object for current updating
// @params [in]: old object for last updating
// @params [in]: time interval from last updating
void MeasureDetectBboxCenterPointVelocity(TrackedObjectPtr new_object,
                                const TrackedObjectConstPtr& old_object,
                                const double& time_diff);

// @brief: measure bbox center velocity
// @params [in/out]: new object for current updating
// @params [in]: old object for last updating
// @params [in]: time interval from last updating
void MeasureBboxCenterVelocity(TrackedObjectPtr new_object,
                               const TrackedObjectConstPtr& old_object,
                               const double& time_diff);

// @brief: measure bbox corner velocity
// @params [in/out]: new object for current updating
// @params [in]: old object for last updating
// @params [in]: time interval from last updating
void MeasureBboxCornerVelocity(TrackedObjectPtr new_object,
                               const TrackedObjectConstPtr& old_object,
                               const double& time_diff);

// @brief: measure bbox corner velocity
// @params [in/out]: new object for current updating
// @params [in]: old object for last updating
// @params [in]: time interval from last updating
void MeasureDetectBboxCornerVelocity(TrackedObjectPtr new_object,
                               const TrackedObjectConstPtr& old_object,
                               const double& time_diff);
}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

