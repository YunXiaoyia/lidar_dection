
//
// Created by jie.gong on 22-03-17.
//

#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <array>

#include "lib/interface/base_target_track.h"
#include "simple_track/tracklet.h"

#include "associater/associater.h"
#include "simple_track/id_manager.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

/**
 * @brief SimpleTrack, A algorithm set under Tracking-By-Detection framework.
 */
class SimpleTrack : public BaseTargetTrack {
   public:
    SimpleTrack() = default;
    bool Init(const toml::node_view<const toml::node>& param_node) override;

    bool Track(common::LidarFramePtr& frame_data) override;

   private:
    // void ObjectsAssociation(const std::vector<common::ObjectPtr>& objects, double cur_time,
    //                         AssociationResult& assignments);

    void UpdateTracklectsPool(const AssociationResult& assignments, const double& cur_time);

    void UpdateAssignedTracks(const std::vector<common::ObjectPtr>& objects,
                              const AssociationResult& assignments,
                              const double& current_time);

    void UpdateUnassignedTracks(const AssociationResult& assignments, const double& cur_time);

    void UpdateUnassignedObjects(const AssociationResult& assignments,
                                 const std::vector<common::ObjectPtr>& detected_objects);

    void TransTrackletsToCurrentWorldPose(const Eigen::Isometry3f& tf);

    void CollectTrackingReault(std::vector<common::ObjectPtr>& objects_out, double cur_time);

    void correctYVelocity(const TrackedObjectConstPtr& tracked_obj, Eigen::Vector3f& velo_out);

    void correctCovariance(const TrackedObjectConstPtr& tracked_obj, \
                           float& orientation_covariance,\
                           float& length_covariance, \
                           float& width_covariance);

    void correctBoxHeading(const TrackedObjectConstPtr& tracked_obj, float& heading, float& heading_cov);

   private:
    // Track pool
    std::vector<TrackletPtr> tracklets_;

    // track object matcher
    AssociaterPtr associater_;

    // For ID manager
    IDManagerPtr id_manager_;

    // params
    float fov_ = 80.0f;
    float delta_x_correction_velocity_ = 10.0f;
    float delta_x_fov_down_ = 5.4f;   // 下边界fov

    float tracklet_init_confidence_threshold_ = 0.5;
    float tracklet_init_iou_threshold_ = 0.4;
    float tracklet_init_confidence_iou_threshold_ = 0.6;
};

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
