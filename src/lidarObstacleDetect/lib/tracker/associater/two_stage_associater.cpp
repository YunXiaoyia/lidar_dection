

#include "associater/two_stage_associater.h"
#include <cstddef>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "common/depthink_time.h"
#include "common/lidar_perception_log.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

TwoStageAssociater::TwoStageAssociater() {

    track_object_distance_ = std::make_unique<ObjectDistanceMeasurement>();
}

bool TwoStageAssociater::Init(const toml::node_view<const toml::node>& param_node) {

    if (!param_node["two_stage"].is_table()) {
        TLOG_WARN << "[TwoStageAssociater] Dont't find two_stage association definition!";
        return false;
    }

    auto paras = param_node["two_stage"];
    if (!paras["optimizer"].is_value()) {
        TLOG_WARN << "[TwoStageAssociater] Dont't find optimizer definition! please check param: optimizer."
                     << " use default: hungarian";
    } else {
        auto type = paras["optimizer"].value<std::string>().value();
        std::cout << "type:" << type << std::endl;
        if (type == "hungarian") {
            optimizer_type_ = OptimizerType::HUNGARIAN;
        } else if (type == "greedy") {
            optimizer_type_ = OptimizerType::GREEDY;
        }
    }

    if (!paras["match_dist_thresh"].is_value()) {
        TLOG_WARN << "[TwoStageAssociater] Dont't find match_dist_thresh definition! please check param: match_dist_thresh."
                     << " use default:" << MATCH_DIST_THRESH;
    } else {
        std::cout << "match_dist_thresh:" << MATCH_DIST_THRESH << std::endl;
        MATCH_DIST_THRESH = paras["match_dist_thresh"].value<float>().value();
    }

    return true;
}

void TwoStageAssociater::Match(const std::vector<common::ObjectPtr> &objects,
                               const std::vector<TrackletPtr> &tracks,
                               AssociationResult& assign_result) {
    
    // Check input objects valid
    assign_result.assignments.clear();
    assign_result.unassigned_measurements.clear();
    assign_result.unassigned_tracks.clear();
    if (objects.empty() || tracks.empty()) {
        assign_result.unassigned_measurements.resize(objects.size());
        assign_result.unassigned_tracks.resize(tracks.size());
        std::iota(assign_result.unassigned_measurements.begin(), assign_result.unassigned_measurements.end(), 0);
        std::iota(assign_result.unassigned_tracks.begin(), assign_result.unassigned_tracks.end(), 0);
        return;
    }

    // Stage 1 association, confirmed tracklet first
    AssociationResult stage1_res;
    std::vector<size_t> confirmed_idxs, other_tracklet_idxs;
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (tracks[i]->isConfirmed()) {
            confirmed_idxs.push_back(i);
        } else {
            other_tracklet_idxs.push_back(i);
        }
    }

    std::vector<size_t> candidate_idxs;
    candidate_idxs.resize(objects.size());
    std::iota(candidate_idxs.begin(), candidate_idxs.end(), 0);
    Associate(tracks, confirmed_idxs, objects, candidate_idxs, stage1_res);

    // Stage 2 association, other tracklets
    AssociationResult stage2_res;
    for (const auto& unassign_trk : stage1_res.unassigned_tracks) {
        other_tracklet_idxs.emplace_back(unassign_trk);
    }

    std::vector<size_t> other_candidate_idxs = stage1_res.unassigned_measurements;

    Associate(tracks, other_tracklet_idxs, objects, other_candidate_idxs, stage2_res);

    // Final, combine two stage results
    assign_result = stage2_res;
    assign_result.assignments.insert(assign_result.assignments.end(), 
                                     stage1_res.assignments.begin(), stage1_res.assignments.end());
}

void TwoStageAssociater::Associate(const std::vector<TrackletPtr> &tracks,
                                   const std::vector<size_t> &track_idxs,
                                   const std::vector<common::ObjectPtr> &new_objects,
                                   const std::vector<size_t> &object_idxs,
                                   AssociationResult& assign_result) {

    if (track_idxs.empty() || object_idxs.empty()) {
        assign_result.unassigned_measurements = object_idxs;
        assign_result.unassigned_tracks = track_idxs;
        return;
    }
    
    // Construct associate matrix
    association_mat_.Resize(track_idxs.size(), object_idxs.size());

    for (size_t i = 0; i < track_idxs.size(); ++i) {
        for (size_t j = 0; j < object_idxs.size(); ++j) {
            
            (association_mat_)(i, j) = 
                track_object_distance_->ComputeDistance(new_objects[object_idxs[j]], tracks[track_idxs[i]]);
        }
    }

    // Set optimizer work way, find min or max 
    AssociationResult cur_res;
    if (optimizer_type_ == OptimizerType::HUNGARIAN) {
        hungarian_optimizer_.Match(MATCH_DIST_THRESH, MATCH_DIST_BOUND, 
                                    GatedHungarianMatcher<float>::OptimizeFlag::OPTMIN, 
                                    association_mat_,  
                                    &cur_res.assignments, 
                                    &cur_res.unassigned_tracks, 
                                    &cur_res.unassigned_measurements);
    } else if (optimizer_type_ == OptimizerType::GREEDY) {
        greedy_optimizer_.Match(MATCH_DIST_THRESH, MATCH_DIST_BOUND, 
                                    association_mat_, 
                                    &cur_res.assignments, 
                                    &cur_res.unassigned_tracks, 
                                    &cur_res.unassigned_measurements);
    }

    
    // collect result
    for (const auto& ass : cur_res.assignments) {
        assign_result.assignments.emplace_back(track_idxs[ass.first], object_idxs[ass.second]);
    }

    for (const auto& unassign_track : cur_res.unassigned_tracks) {
        assign_result.unassigned_tracks.emplace_back(track_idxs[unassign_track]);
    }

    for (const auto& unassign_obj : cur_res.unassigned_measurements) {
        assign_result.unassigned_measurements.emplace_back(object_idxs[unassign_obj]);
    }
}

void TwoStageAssociater::ComputeAssociateMatrix(
    const std::vector<TrackletPtr> &tracks,
    const std::vector<common::ObjectPtr> &new_objects,
    SecureMat<float> *association_mat) {
    for (size_t i = 0; i < tracks.size(); ++i) {
        for (size_t j = 0; j < new_objects.size(); ++j) {
            
            (*association_mat)(i, j) = 
                track_object_distance_->ComputeDistance(new_objects[j], tracks[i]);
            // std::cout << "track_id:" << tracks[i]->track_id() << "  object_id:" << new_objects[j]->id << "  dis:" << (*association_mat)(i, j) << std::endl;
        }
    }
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
