
//
// Created by jie.gong on 22-06-30.
//

#pragma once

#include <string>

#include "common/tracked_object.h"
#include "common/registerer.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

struct TrackFilterInitOptions {
    TrackedObjectPtr tracked_object;

    TrackFilterInitOptions() {}
    TrackFilterInitOptions(TrackedObjectPtr& tracked_obj) :
        tracked_object(tracked_obj) {}
};

class BaseTrackFilter {
 public:
    BaseTrackFilter() = default;
    virtual ~BaseTrackFilter() = default;

    virtual bool Init(
        const TrackFilterInitOptions& options = TrackFilterInitOptions()) = 0;

    // @brief: interface for updating filter with object
    // @params [in]: options for updating
    // @params [in]: track data, not include new object
    // @params [in/out]: new object for updating
    virtual void UpdateWithObject(const TrackedObjectConstPtr& latest_object, 
                                    TrackedObjectPtr& new_object) = 0;

    // @brief: interface for updating filter without object
    // @params [in]: options for updating
    // @params [in]: current timestamp
    // @params [in/out]: track data to be updated
    virtual void UpdateWithoutObject(const TrackedObjectConstPtr& latest_object, 
                                    TrackedObjectPtr& new_object) = 0;
    
    virtual void TransState(const Eigen::Isometry3f& tf) {};

    virtual std::string Name() const = 0;

};  // class BaseTrackFilter

PERCEPTION_REGISTER_REGISTERER(BaseTrackFilter);  // 注册基类
#define PERCEPTION_REGISTER_TRACK_FILTER(name) \
  PERCEPTION_REGISTER_CLASS(highway::perception::track::BaseTrackFilter, name)

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
