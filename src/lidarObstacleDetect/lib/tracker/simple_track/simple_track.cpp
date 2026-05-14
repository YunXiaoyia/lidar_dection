
//
// Created by jie.gong on 22-02-17.
//

#include <math.h>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <sstream>

// 临时定义 TLOG_INFO 和 TLOG_WARN 宏
#ifndef TLOG_INFO
#define TLOG_INFO std::cout << "[INFO] "
#endif
#ifndef TLOG_WARN
#define TLOG_WARN std::cout << "[WARN] "
#endif

// 临时定义 utils 命名空间和 inFOV 函数
namespace utils {
    inline bool inFOV(float x, float y, float fov_degrees, float delta_x_corner_shift) {
        float fov_rad = fov_degrees * M_PI / 180.0f;
        float tan_half_fov = std::tan(fov_rad / 2.0f);
        
        if (x > -delta_x_corner_shift) {
            return std::abs(y) < (x + delta_x_corner_shift) * tan_half_fov;
        }
        return false;
    }
}

#include "simple_track.h"

#include "common/debug_singleton.h"
#include "common/depthink_time.h"
#include "lib/utils/geometry.hpp"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

////////////////////////////////////////
// For debug, print object info
////////////////////////////////////////

void printTrackedResult(const TrackletPtr& track_info) {

    const auto& tracked_obj = track_info->GetLatestTrackedObject();

    std::cout << "track id:" << track_info->track_id() << " age:" << track_info->track_age() << " "
        << " tracked size:" << track_info->history_tracked_objects().size() << " "
        << " detected size:" << track_info->history_detected_objects().size() << " "
        << " size:" << tracked_obj->output_size[0] << " "
        << " " << tracked_obj->output_size[1] << " " << tracked_obj->output_size[2]
        << "pos:" << tracked_obj->output_center[0] << " " << tracked_obj->output_center[1]
        << " " << tracked_obj->output_center[2]
        << " output_velocity:" << tracked_obj->output_velocity[0] << " "
        << tracked_obj->output_velocity[1] << " "
        << tracked_obj->output_velocity[2] << std::endl;
}

void printAssignmentsResult(const AssociationResult& assignments,
                            const std::vector<TrackletPtr>& tracklets,
                            const std::vector<common::ObjectPtr>& detected_objects) {
    {
        std::stringstream ss;
        ss << "Assigned result:";
        for (const auto& t : assignments.assignments) {
            ss << tracklets[t.first]->track_id() << "  " << detected_objects[t.second]->detect_id;
        }
        TLOG_INFO << ss.str();
    }

    {
        std::stringstream ss;
        ss << "unassiged tracks: ";
        for (const auto& t : assignments.unassigned_tracks) {
            ss << tracklets[t]->track_id() << " ";
        }
        TLOG_INFO << ss.str();
    }

    {
        std::stringstream ss;
        ss << "unassiged objects: ";
        for (const auto& t : assignments.unassigned_measurements) {
            ss << detected_objects[t]->detect_id << " ";
        }
        TLOG_INFO << ss.str();
    }
}

////////////////////////////////////////
// Class function
////////////////////////////////////////

bool SimpleTrack::Init(const toml::node_view<const toml::node>& param_node) {

    try {
        auto params = param_node["simple_track"];
        delta_x_correction_velocity_ = params["delta_x_correction_velocity"].value<float>().value();
        delta_x_fov_down_ = params["delta_x_fov_down"].value<float>().value();
        fov_ = params["fov"].value<float>().value();
        tracklet_init_confidence_threshold_ = params["tracklet_init_confidence_threshold"].value<float>().value();
        tracklet_init_iou_threshold_ = params["tracklet_init_iou_threshold"].value<float>().value();
        tracklet_init_confidence_iou_threshold_ = params["tracklet_init_confidence_iou_threshold"].value<float>().value();

        // Init ID manager
        id_manager_ = std::make_shared<IDManager>();
        if (params["id_manager"].is_table()) {
            id_manager_->Init(params["id_manager"]);
        }

        // Init associater
        associater_ = AssociaterFactory::MakeAssociater(params);
        if (associater_ == nullptr) {
            TLOG_WARN << "[tracker] Associater init failed!";
            return false;
        }

        // Init tracklet params
        if (!Tracklet::InitStaticParams(params)) {
            TLOG_WARN << "[tracker] Tracklet init failed!";
            return false;
        }

        TLOG_INFO << "[tracker] : init SimpleTrack Successfully";
        return true;
    } catch (const std::exception& e) {
        TLOG_WARN << "[tracker] : init SimpleTrack Unsuccessfully : " << e.what();
        return false;
    }
}

void SimpleTrack::UpdateTracklectsPool(const AssociationResult& assignments, const double& cur_time) {

    size_t valid_num = 0;
    for (size_t idx = 0; idx < tracklets_.size(); ++idx) {

        if (!tracklets_[idx]->isAlive(cur_time)) {
            id_manager_->id_recycled(tracklets_[idx]->track_id());
            continue;
        }

        if (tracklets_[idx]->isDeleted()) {
            id_manager_->id_recycled(tracklets_[idx]->track_id());
            continue;
        }

        if (valid_num != idx) {
            tracklets_[valid_num] = tracklets_[idx];
        }
        valid_num++;
    }

    tracklets_.resize(valid_num);
}

void SimpleTrack::UpdateUnassignedTracks(const AssociationResult& assignments, const double& cur_time) {
    for (const size_t& idx : assignments.unassigned_tracks) {
        tracklets_[idx]->updateWithoutObject(cur_time);
    }
}

void SimpleTrack::UpdateAssignedTracks(const std::vector<common::ObjectPtr>& objects,
                                          const AssociationResult& assignments, const double& cur_time) {

    for (const auto& ass : assignments.assignments) {
        if(tracklets_[ass.first]->track_age()==1){
            const auto& det = objects[ass.second];
            float iou_ref = 0.7;
            // 临时使用默认值，因为 bbox 没有 confidence 和 iou 成员
            float confidence = 0.9f;  // 默认置信度
            float iou = 0.8f;  // 默认 IOU
            float conf_iou =  std::pow(confidence, (1.0-iou_ref)) * std::pow(iou , iou_ref);

            if(confidence < tracklet_init_confidence_threshold_
                || iou < tracklet_init_iou_threshold_
                || conf_iou <tracklet_init_confidence_iou_threshold_){


                    tracklets_[ass.first]->updateWithoutObject(cur_time);
                    continue;
            }
        }

        tracklets_[ass.first]->updateWithObject(objects[ass.second]);

        if (tracklets_[ass.first]->isSkeptical() &&
            tracklets_[ass.first]->consecutive_visible_count() > tracklets_[ass.first]->getMinContinuousDetNumThreshold()) {
            tracklets_[ass.first]->set_status(TrackletStatus::CONFIRMED);
            tracklets_[ass.first]->set_track_id(id_manager_->get_id());
        }
    }
}

void SimpleTrack::UpdateUnassignedObjects(const AssociationResult& assignments,
                            const std::vector<common::ObjectPtr>& detected_objects) {

    // perf by lgj
    // 对于新增的目标，其置信度和IOU阈值要比较大才init

    for (const auto idx : assignments.unassigned_measurements) {
        TrackletPtr tracklet(new Tracklet);
        TrackletInitParams init_param;
        init_param.fov_theta = fov_;
        init_param.x_direction_offset = delta_x_correction_velocity_;
        float iou_ref = 0.7;
        // 临时使用默认值，因为 bbox 没有 confidence 和 iou 成员
        float confidence = 0.9f;  // 默认置信度
        float iou = 0.8f;  // 默认 IOU
        float conf_iou =  std::pow(confidence, (1.0-iou_ref)) * std::pow(iou , iou_ref);

        if(confidence < tracklet_init_confidence_threshold_
        || iou < tracklet_init_iou_threshold_
        || conf_iou <tracklet_init_confidence_iou_threshold_){
            continue;
        }

        tracklet->Init(detected_objects[idx]->timestamp, detected_objects[idx], init_param);
        tracklet->set_status(TrackletStatus::SKEPTICAL);

        tracklets_.push_back(tracklet);
    }
}

void SimpleTrack::correctYVelocity(const TrackedObjectConstPtr& tracked_obj, Eigen::Vector3f& velo_out) {

      if (!utils::inFOV(tracked_obj->output_center.x(),tracked_obj->output_center.y(), fov_,
                                                    delta_x_correction_velocity_)) {
            if (tracked_obj->output_center.y() > 1.0f && tracked_obj->output_center.y() < 4.5f && tracked_obj->output_velocity.y() > -1.5f &&
                tracked_obj->output_velocity.y() < 0.0f) {
                velo_out[1] = 0.0f;
            } else if (tracked_obj->output_center.y() < -1.0f && tracked_obj->output_center.y() > -4.5f &&
                       tracked_obj->output_velocity.y() < 1.5f && tracked_obj->output_velocity.y() > 0.0f) {
                velo_out[1] = 0.0f;
            }
        }
}


void SimpleTrack::correctCovariance(const TrackedObjectConstPtr& tracked_obj, \
                                    float& orientation_covariance, \
                                    float& length_covariance, \
                                    float& width_covariance) {
    const auto& box_center = tracked_obj->output_center;

    // delta_x_correction_velocity_ = 12.0f; delta_x_fov_down_ = 5.4f; fov_ =12.0f;
    const static float delta_distance = fabs(delta_x_correction_velocity_ - delta_x_fov_down_);
    const static float ori_cova_max = powf(M_PI_2, 2);
    const static float ori_cova_min = powf((M_PI / 16), 2);
    const static float resolution_ori_cova = (ori_cova_max - ori_cova_min) / delta_distance;
    const static float k = fabs(tan(fov_ * M_PI / 360.0));
    //
    if(!utils::inFOV(box_center.x(), box_center.y(), fov_, delta_x_correction_velocity_)){
        float delta_x = fabs(fabs(box_center.y()) / k + delta_x_correction_velocity_ - box_center.x());
        float covariance = delta_x * resolution_ori_cova + ori_cova_min;
        orientation_covariance = covariance;
        length_covariance      = covariance;
        width_covariance       = covariance;
    }
    else{
        orientation_covariance = ori_cova_min;
        length_covariance      = ori_cova_min;
        width_covariance       = ori_cova_min;
    }
}

// void SimpleTrack::CollectTrackingReault(std::vector<common::ObjectPtr>& objects_out, double cur_time) {
//     objects_out.clear();

//     for (const auto& track_info : tracklets_) {

//         if (track_info->isSkeptical() || track_info->isDeleted()) {
//             continue;
//         }

//         // if (track_info->consecutive_invisible_count() > 0) {
//         //     continue;
//         // }

//         // TLOG_INFO << "track id:" << track_info->track_id() << " age:" << track_info->track_age();

//         const auto tracked_obj = track_info->GetOutTrackedObject();
//         const auto detected_obj = track_info->GetLatestDetectedObject();

//         if (tracked_obj == nullptr || detected_obj == nullptr) {
//             continue;
//         }

//         // printTrackedResult(track_info);

//         common::ObjectPtr obj_out = std::make_shared<common::Object>();
//         obj_out->timestamp = tracked_obj->timestamp;
//         // 临时使用默认值，因为 Object 没有 sensor_id 成员
//         // obj_out->sensor_id = 0;  // 默认传感器ID (sensor_id 成员不存在)
//         obj_out->detect_id = detected_obj->detect_id;   // 导出与其关联的检测id  debug用
//         obj_out->track_id = track_info->track_id();
//         obj_out->age = track_info->track_age();
//         obj_out->type = tracked_obj->output_type;
//         obj_out->velocity = tracked_obj->output_velocity;
//         obj_out->state_covariance = tracked_obj->output_state_covariance;
//         if(obj_out->age < 10){
//             obj_out->state_covariance.array() *= 10.0f;
//         }
//         else if(obj_out->age < 20){
//             obj_out->state_covariance.array() *= 5.0f;
//         }

//         // TLOG_INFO << " track id:" << obj_out->track_id << "  state covariance:\n" << obj_out->state_covariance;

//         // // Tricks, Y-AXIS velocity calibration
//         // correctYVelocity(tracked_obj, obj_out->velocity);
        
//         // // 临时使用默认值，因为 Object 没有这些协方差成员


//         // [修改后] 将其替换为如下代码：
//         // ==========================================
//         // [New] 照搬 LShapeTrack 的速度处理逻辑
//         // ==========================================
        
//         // 1. 速度限制 > 50m/s (180km/h) 则丢弃该目标
//         // 注意：需确保包含了 <cmath> 或使用 Eigen 的 norm
//         float velocity_abs = std::hypot(obj_out->velocity.x(), obj_out->velocity.y());
//         if (velocity_abs > 50.0f) {
//             continue; 
//         }

//         // 2. 近 FOV 区内/外 速度校正回归
//         // 使用 SimpleTrack.cpp 头部定义的本地 utils::inFOV 函数
//         // 逻辑与 LShapeTrack 完全一致：如果在 FOV 边缘盲区且横向速度较小，则强制归零横向速度
//         if (!utils::inFOV(obj_out->bbox.center.x(), obj_out->bbox.center.y(), fov_,
//                                                     delta_x_correction_velocity_)) {
//             if (obj_out->bbox.center.y() > 1.0f && obj_out->bbox.center.y() < 4.5f && obj_out->velocity.y() > -1.5f &&
//                 obj_out->velocity.y() < 0.0f) {
//                 obj_out->velocity.y() = 0.0f;
//             } else if (obj_out->bbox.center.y() < -1.0f && obj_out->bbox.center.y() > -4.5f &&
//                        obj_out->velocity.y() < 1.5f && obj_out->velocity.y() > 0.0f) {
//                 obj_out->velocity.y() = 0.0f;
//             }
//         }
        
//         // ==========================================
//         // [End]
//         // ==========================================
//         float orientation_covariance = 0.1f;
//         float length_covariance = 0.1f;
//         float width_covariance = 0.1f;
//         correctCovariance(tracked_obj, orientation_covariance, length_covariance, width_covariance);
        
//         // 使用速度方向、校正box-heading
//         float box_heading = tracked_obj->output_theta;
//         correctBoxHeading(tracked_obj, box_heading, orientation_covariance);

//         // Copy boundingbox info
//         // 确保 corners2d 矩阵已正确初始化
//         if (obj_out->bbox.corners2d.rows() != 2 || obj_out->bbox.corners2d.cols() != 4) {
//             obj_out->bbox.corners2d.resize(2, 4);
//         }
        
//         obj_out->bbox.corners2d.col(0) << tracked_obj->output_corners[0].x(), tracked_obj->output_corners[0].y();
//         obj_out->bbox.corners2d.col(1) << tracked_obj->output_corners[1].x(), tracked_obj->output_corners[1].y();
//         obj_out->bbox.corners2d.col(2) << tracked_obj->output_corners[2].x(), tracked_obj->output_corners[2].y();
//         obj_out->bbox.corners2d.col(3) << tracked_obj->output_corners[3].x(), tracked_obj->output_corners[3].y();
//         obj_out->bbox.size = tracked_obj->output_size;
//         obj_out->bbox.center = tracked_obj->output_center;
//         obj_out->bbox.theta = box_heading;
        
//         // 临时使用默认值，因为 bbox 没有 confidence 成员
//         // obj_out->bbox.confidence = 0.9f;  // 默认置信度 (confidence 成员不存在)
//         // obj_out->bbox.type_probs = detected_obj->bbox.type_probs;  // 不存在此成员

//         // TODO, polygon - 临时注释掉，因为 Object 没有 polygon 成员
//         // obj_out->polygon.resize(4);
//         // for(int i = 0; i < 4; i++){
//         //     obj_out->polygon[i].x() = obj_out->bbox.corners2d(0, i);
//         //     obj_out->polygon[i].y() = obj_out->bbox.corners2d(1, i);
//         // }

//         objects_out.emplace_back(obj_out);
//     }
// }
void SimpleTrack::CollectTrackingReault(std::vector<common::ObjectPtr>& objects_out, double cur_time) {
    objects_out.clear();

    for (const auto& track_info : tracklets_) {

        if (track_info->isSkeptical() || track_info->isDeleted()) {
            continue;
        }

        const auto tracked_obj = track_info->GetOutTrackedObject();
        const auto detected_obj = track_info->GetLatestDetectedObject();

        if (tracked_obj == nullptr || detected_obj == nullptr) {
            continue;
        }

        common::ObjectPtr obj_out = std::make_shared<common::Object>();
        obj_out->timestamp = tracked_obj->timestamp;
        obj_out->detect_id = detected_obj->detect_id;   
        obj_out->track_id = track_info->track_id();
        obj_out->age = track_info->track_age();
        obj_out->type = tracked_obj->output_type;
        obj_out->velocity = tracked_obj->output_velocity; // 获取原始跟踪速度
        obj_out->state_covariance = tracked_obj->output_state_covariance;
        
        // 协方差放大逻辑 (保留 SimpleTrack 原有逻辑)
        if(obj_out->age < 10){
            obj_out->state_covariance.array() *= 10.0f;
        }
        else if(obj_out->age < 20){
            obj_out->state_covariance.array() *= 5.0f;
        }

        // =========================================================
        // [修改] 照搬 LShapeTrack 的速度处理逻辑 (替换原 correctYVelocity)
        // =========================================================

        // 1. 速度限制: > 50m/s (180km/h) 则丢弃
        // 注意：需确保包含 <cmath>，或者使用 hypot/Eigen norm
        float velocity_abs = std::hypot(obj_out->velocity.x(), obj_out->velocity.y());
        if (velocity_abs > 50.0f) {
            continue; 
        }

        // 2. 近 FOV 区内/外 速度校正回归
        // 依赖 utils::inFOV (需确保 simple_track.cpp 头部已定义该辅助函数)
        // 依赖 fov_, delta_x_correction_velocity_ (需确保 Init 已正确加载这些参数)
        if (!utils::inFOV(obj_out->bbox.center.x(), obj_out->bbox.center.y(), fov_, delta_x_correction_velocity_)) {
            // 如果在左侧盲区，且有向右的微小速度，强制归零 Vy
            if (obj_out->bbox.center.y() > 1.0f && obj_out->bbox.center.y() < 4.5f && 
                obj_out->velocity.y() > -1.5f && obj_out->velocity.y() < 0.0f) {
                obj_out->velocity.y() = 0.0f;
            } 
            // 如果在右侧盲区，且有向左的微小速度，强制归零 Vy
            else if (obj_out->bbox.center.y() < -1.0f && obj_out->bbox.center.y() > -4.5f &&
                     obj_out->velocity.y() < 1.5f && obj_out->velocity.y() > 0.0f) {
                obj_out->velocity.y() = 0.0f;
            }
        }
        // =========================================================

        // 临时使用默认值
        float orientation_covariance = 0.1f;
        float length_covariance = 0.1f;
        float width_covariance = 0.1f;
        
        // 保留 SimpleTrack 的协方差修正
        correctCovariance(tracked_obj, orientation_covariance, length_covariance, width_covariance);
        
        // 保留 SimpleTrack 的航向角修正
        float box_heading = tracked_obj->output_theta;
        correctBoxHeading(tracked_obj, box_heading, orientation_covariance);

        // Copy boundingbox info
        if (obj_out->bbox.corners2d.rows() != 2 || obj_out->bbox.corners2d.cols() != 4) {
            obj_out->bbox.corners2d.resize(2, 4);
        }
        
        obj_out->bbox.corners2d.col(0) << tracked_obj->output_corners[0].x(), tracked_obj->output_corners[0].y();
        obj_out->bbox.corners2d.col(1) << tracked_obj->output_corners[1].x(), tracked_obj->output_corners[1].y();
        obj_out->bbox.corners2d.col(2) << tracked_obj->output_corners[2].x(), tracked_obj->output_corners[2].y();
        obj_out->bbox.corners2d.col(3) << tracked_obj->output_corners[3].x(), tracked_obj->output_corners[3].y();
        obj_out->bbox.size = tracked_obj->output_size;
        obj_out->bbox.center = tracked_obj->output_center;
        obj_out->bbox.theta = box_heading;
        
        objects_out.emplace_back(obj_out);
    }
}

void SimpleTrack::correctBoxHeading(const TrackedObjectConstPtr& tracked_obj, float& heading, float& heading_cov){
    float velocity_abs = tracked_obj->output_velocity.norm();
    float velo_heading = atan2(tracked_obj->output_velocity.y(), tracked_obj->output_velocity.x());
    Eigen::Vector2f direction_velo, direction_box;
    direction_velo << cos(velo_heading), sin(velo_heading);
    direction_box  << cos(tracked_obj->output_theta), sin(tracked_obj->output_theta);
    // 速度较大时
    if(velocity_abs >= 10.0f){
        if (direction_velo.dot(direction_box) < 0) {
            direction_box = (-1) * direction_box;
            heading = std::atan2(direction_box.y(), direction_box.x());
        }
    }
    else{
        // 速度较小时，
        if (direction_velo.dot(direction_box) < 0) {
            heading_cov *= 2.0;
         }
    }
}


void SimpleTrack::TransTrackletsToCurrentWorldPose(const Eigen::Isometry3f& tf) {
    for (const auto& tracklet : tracklets_) {
        tracklet->TransformToCurrentFrame(tf);
    }
}

bool SimpleTrack::Track(common::LidarFramePtr& frame_data) {

    std::vector<common::ObjectPtr>& detected_objects = frame_data->detected_objects;
    // for (auto& obj : detected_objects) {
    //     obj->timestamp = frame_data->timestamp;
    // }
    const auto& current_frame_time = frame_data->timestamp;

    // TLOG_INFO << "tracker------------------------------------> tracklet:"
    //           << tracklets_.size() << "  object:" << detected_objects.size();

    // PERF_BLOCK_START();
    // 1. add global offset to pose (only when no track exists)
    TransTrackletsToCurrentWorldPose(frame_data->tf);
    // PERF_BLOCK_END("TRANS COST");

    // Associate objects and tracks
    AssociationResult assignments;
    associater_->Match(detected_objects, tracklets_, assignments);
    // PERF_BLOCK_END("MATCH COST");

    // printAssignmentsResult(assignments, tracklets_, detected_objects);

    // update assigned tracks
    UpdateAssignedTracks(detected_objects, assignments, current_frame_time);
    // PERF_BLOCK_END("ASSIGN COST");
    // Update unassigned objects
    UpdateUnassignedObjects(assignments, detected_objects);

    // update unassigned tracks
    UpdateUnassignedTracks(assignments, current_frame_time);

    // Update tracklets pool
    UpdateTracklectsPool(assignments, current_frame_time);

    // Collect tracking result
    CollectTrackingReault(frame_data->tracked_objects, current_frame_time);
    // PERF_BLOCK_END("OTHER COST");

    return true;
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
