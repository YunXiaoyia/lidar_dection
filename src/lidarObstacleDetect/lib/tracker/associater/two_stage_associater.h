//
// Created by jie.gong on 22-06-30.
//
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include "common/base.h"

#include "associater/distance_measurement/object_distance_measurement.h"
#include "associater/associater.h"
#include "associater/hungarian/gated_hungarian_bigraph_matcher.h"
#include "associater/greedy/greedy_matcher.h"
#include "secure_matrix.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


enum class OptimizerType : std::uint8_t {
    UNKNOWN,
    HUNGARIAN,
    GREEDY
};

class TwoStageAssociater : public BaseAssociater {
  
private:
    /// ////////////////////
    /// Hyper Parameters
    /// ////////////////////

    // Limit maxmum associate distance
    float MATCH_DIST_THRESH = 4.0;

    // Limit associate boundary
    float MATCH_DIST_BOUND = 100.0;

    // Limit maxmum center point associate distance
    float ASSOCIATE_CENTER_DIST_THRESH = 30.0;

public:
    TwoStageAssociater();
    ~TwoStageAssociater() = default;

    bool Init(const toml::node_view<const toml::node>& param_node) override;

    void Match(const std::vector<common::ObjectPtr> &objects,
               const std::vector<TrackletPtr> &tracks,
               AssociationResult& assign_result) override;

    std::string Name() const override { return "TwoStageAssociater"; };

protected:
    // @brief: compute association matrix
    // @params [in]: maintained tracks for matching
    // @params [in]: new detected objects for matching
    // @params [out]: matrix of association distance
    void ComputeAssociateMatrix(const std::vector<TrackletPtr> &tracks,
                                const std::vector<common::ObjectPtr> &new_objects,
                                SecureMat<float> *association_mat);

    void Associate(const std::vector<TrackletPtr> &tracks,
                   const std::vector<size_t> &track_idxs,
                   const std::vector<common::ObjectPtr> &new_objects,
                   const std::vector<size_t> &object_idxs,
                   AssociationResult& assign_results);

protected:
    std::unique_ptr<ObjectDistanceMeasurement> track_object_distance_;

    // For hungarian optimizer solve
    GatedHungarianMatcher<float> hungarian_optimizer_;

    // For greedy optimizer solve
    GreedyMatcher greedy_optimizer_;

    // Optimizer type
    OptimizerType optimizer_type_ = OptimizerType::HUNGARIAN;

    /* global costs matrix */
    SecureMat<float> association_mat_;

};  // class MlfTrackObjectMatcher



}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

