#pragma once

#include "common/base_track_filter.h"
#include "common/tracked_object.h"

#include "../../lidarnetsdk/src/common/toml.hpp"
#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


class MovingAverageShapeFilter : public BaseTrackFilter {

public:
    MovingAverageShapeFilter() = default;
    virtual ~MovingAverageShapeFilter() = default;

    /**
     * @brief init static params
     *
     * @param param_node params handle
     * @return true if success or false if there are something failed
     */
    static bool InitStaticParams(const toml::node_view<const toml::node>& param_node);

    bool Init(
      const TrackFilterInitOptions& options = TrackFilterInitOptions()) override;

    // @brief: updating shape filter with object
    // @params [in]: options for updating
    // @params [in]: track data, not include new object
    // @params [in/out]: new object for updating
    void UpdateWithObject(const TrackedObjectConstPtr& latest_object,
                          TrackedObjectPtr& new_object) override;

    // @brief: updating shape filter without object
    // @params [in]: options for updating
    // @params [in]: current timestamp
    // @params [in/out]: track data to be updated
    void UpdateWithoutObject(const TrackedObjectConstPtr& latest_object,
                             TrackedObjectPtr& track_data) override;

    std::string Name() const override { return "MovingAverageShapeFilter"; }

private:

    /// ////////////////////
    /// Hyper Parameters
    /// ////////////////////

    // Moving average coefficient of heading angle
    static float AnglekMovingAverage;

    // Moving average coefficient of bbox size
    static float SizekMovingAverage;

    // params
    static float fov_;
    static float delta_x_corner_shift_;

};  // class MovingAverageShapeFilter

PERCEPTION_REGISTER_TRACK_FILTER(MovingAverageShapeFilter);
}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
