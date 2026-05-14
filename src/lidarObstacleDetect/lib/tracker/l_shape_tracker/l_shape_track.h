
//
// Created by jie.gong on 22-02-17.
//

#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <array>

#include "lib/interface/base_target_track.h"

#include "obj_tracked_3d.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

/**
 * @brief LShapeTrack 基于L-Shape的目标跟踪
 */
class LShapeTrack : public BaseTargetTrack {

public:
    LShapeTrack() = default;
    bool Init(const toml::node_view<const toml::node>& param_node) override;

    bool Track(common::LidarFramePtr& frame_data) override;

private:

    void nms();

    void associateProposalToTrack(const std::vector<common::ObjectPtr>& propVec, const double& time_stamp,
                                    std::vector<int>& assignments);

    void ObjectsAssociation(const std::vector<common::ObjectPtr>& objects, double cur_time,
                            std::vector<int>& assignments);

    // update assigned tracks
    void UpdateUnassignedTracks(const std::vector<int>& assignments);

    void UpdateAssignedTracks(const std::vector<common::ObjectPtr>& objects, double cur_time,
                              const std::vector<int>& assignments);

    void UpdateUnassignedObjects(const std::vector<int>& assignments,
                                 const std::vector<common::ObjectPtr>& detected_objects);

    void CollectTrackingReault(std::vector<common::ObjectPtr>& objects_out);

   private:
    std::vector<ObjTracked3d> tracklets_;

    std::deque<uint16_t> freshIdPool_;

    const double EPSILON_TIME = 1e-3;
    const double DEFAULT_FPS = 0.1;

    float fov_ = 80.0f;
    float delta_x_correction_velocity_ = 10.0f;
};

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

