
//
// Created by jie.gong on 22-02-17.
//

#include "l_shape_track.h"

#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "association/HungarianAlg.h"
#include "meaurement/distance_collection.hpp"
#include "common/debug_singleton.h"
#include "lib/utils/geometry.hpp"
#include "common/lidar_perception_log.h"
#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

// 255 ID池可能不够用
bool LShapeTrack::Init(const toml::node_view<const toml::node>& param_node) {
    try {
        delta_x_correction_velocity_ = param_node["l_shape"]["delta_x_correction_velocity"].value<float>().value();
        fov_ = param_node["l_shape"]["fov"].value<float>().value();

        for (uint16_t i = 0; i < 255; ++i) {
            freshIdPool_.push_back(i);
        }
        TLOG_INFO << "[tracker] : init LShapeTrack Successfully";
        return true;
    } catch (const std::exception& e) {
        TLOG_WARN << "[tracker] : init LShapeTrack Unsuccessfully : " << e.what();
        return false;
    }
}

void LShapeTrack::ObjectsAssociation(const std::vector<common::ObjectPtr>& objects, double cur_time,
                                     std::vector<int>& assignments) {
    assignments.clear();
    size_t sz_detect = objects.size();
    size_t sz_track = tracklets_.size();
    std::vector<std::vector<double>> costs(sz_track, std::vector<double>(sz_detect));

    constexpr float max_distance_threshold = 16.0f;
    for (size_t i = 0; i < sz_track; ++i) {
        for (size_t j = 0; j < sz_detect; ++j) {
            double dt = cur_time - tracklets_[i].last_detect_obj_->timestamp;

            if (dt < 0) {
                costs[i][j] = 1.0;
            } else {
                auto& track = tracklets_[i].corners_;
                auto& detect = objects[j]->bbox.corners2d;

                // TLOG_INFO << "Track id:" << tracklets_[i].track_id_ << "  detect_id:" << objects[j]->id << "  dt:" << dt;

                float min_distance_square = FLT_MAX;
                for (size_t m = 0; m < track.size(); ++m) {
                    for (int n = 0; n < detect.cols(); ++n) {
                        min_distance_square = std::min(min_distance_square, (track[m].topRows(2) - detect.col(n)).squaredNorm());
                    }
                }
                /// iou distance
                float iou = getIOUDistance(tracklets_[i].corners_, objects[j]->bbox.corners2d, tracklets_[i].size_,
                                           objects[j]->bbox.size);

                // auto&& log = COMPACT_GOOGLE_LOG_INFO;
                // log.stream() << "iou:" << iou << "  dis_square:" << min_distance_square;

                if (iou < 0.05 && min_distance_square > max_distance_threshold) {
                    costs[i][j] = 1.0;
                    continue;
                }

                /// point number distance
                float point_num_distance =
                    pointNumDistance(tracklets_[i].last_detect_obj_->points->size(), objects[j]->points->size());

                // log.stream() << "  point_num_distance:" << point_num_distance;
                if (point_num_distance > 0.6f) {
                    costs[i][j] = 1.0;
                    continue;
                }

                /// bbox_size distance
                //                float bbox_size_distance = bboxSizeDistance(track_vec[i].extent,
                //                objects[j].bbox.extent);

                double cx_old = tracklets_[i].center_(0) - 0.5 * tracklets_[i].size_.x();
                double cy_old = tracklets_[i].center_(1);
                double cx_new = objects[j]->bbox.center(0) - 0.5 * objects[j]->bbox.size.x();
                double cy_new = objects[j]->bbox.center(1);
                double dx = cx_new - cx_old;
                double dy = cy_new - cy_old;
                double velo = sqrt(dx * dx + dy * dy) / dt;

                // 速度限制144km/h,即40m/s
                double cost = std::max(0.0, std::min(1.0, velo / 40.0));

                // log.stream() << "  cost:" << cost;

                if (cost > 0.99) {
                    costs[i][j] = 1.0;
                } else {
                    costs[i][j] = cost;
                }
            }
        }
    }

    if (costs.size() > 0) {
        AssignmentProblemSolver solver;
        solver.Solve(costs, assignments, AssignmentProblemSolver::optimal);

        for (size_t i = 0; i < assignments.size(); ++i) {
            if (assignments[i] == -1 || costs[i][assignments[i]] >= 0.99) {
                assignments[i] = -1;
            }
        }
    }
}

void LShapeTrack::UpdateUnassignedTracks(const std::vector<int>& assignments) {
    for (size_t i = 0; i < assignments.size(); ++i) {
        if (assignments[i] == -1) {
            tracklets_[i].consecutive_lost_ += 1;
            tracklets_[i].life_time_ -= 1;
        }
    }
}

void LShapeTrack::nms() {
    for (size_t i = 0; i < tracklets_.size(); ++i) {
        if (tracklets_[i].consecutive_lost_ >= 5) {
            continue;
        }

        for (size_t j = i + 1; j < tracklets_.size(); ++j) {
            if (tracklets_[j].consecutive_lost_ >= 5) {
                continue;
            }

            float iou = getIOUDistance(tracklets_[i].corners_, tracklets_[j].corners_, tracklets_[i].size_, tracklets_[j].size_);

            if (iou > 0.2) {
                if (tracklets_[i].life_time_ > tracklets_[j].life_time_) {
                    tracklets_[j].consecutive_lost_ = 5;
                } else {
                    tracklets_[i].consecutive_lost_ = 5;
                }
            }
        }
    }
}

void LShapeTrack::UpdateAssignedTracks(const std::vector<common::ObjectPtr>& objects, double cur_time,
                                       const std::vector<int>& assignments) {
    for (size_t i = 0; i < assignments.size(); ++i) {
        if (assignments[i] != -1) {
            // double dt = cur_time - tracklets_[i].last_detect_obj_->timestamp;
            tracklets_[i].update(objects[assignments[i]], cur_time);
        }
    }

    // NMS filter
    nms();

    /// 管理生命周期
    for (auto& t : tracklets_) {
        if (t.dieout()) {
            freshIdPool_.push_back(t.track_id_);
        }
    }
    tracklets_.erase(std::remove_if(tracklets_.begin(), tracklets_.end(), [](ObjTracked3d& t) { return t.dieout(); }),
                     tracklets_.end());
}

void LShapeTrack::UpdateUnassignedObjects(const std::vector<int>& assignments,
                                          const std::vector<common::ObjectPtr>& detected_objects) {
    for (size_t j = 0; j < detected_objects.size(); ++j) {
        auto iter = find(assignments.begin(), assignments.end(), j);
        if (iter == assignments.end()) {
            ObjTracked3d track;
            track.life_time_ = 1;
            track.consecutive_lost_ = 0;
            track.track_id_ = freshIdPool_.front();

            if (freshIdPool_.empty()) {
                track.track_id_ = 255;
            } else {
                freshIdPool_.pop_front();
            }

            track.last_detect_obj_ = detected_objects[j];
            // Lshpae特征
            double x_corner = detected_objects[j]->lshape_box.reference_point(0);
            double y_corner = detected_objects[j]->lshape_box.reference_point(1);
            double L1 = detected_objects[j]->lshape_box.l_shape(0);
            double L2 = detected_objects[j]->lshape_box.l_shape(1);
            double thetaL1 = detected_objects[j]->lshape_box.l_shape(2);
            // double dt = 0.099;  // 影响kalamn中的参数矩阵 error: unused variable

            // 初始化
            LshapeFilter l_shape_filt(x_corner, y_corner, L1, L2, normalize_angle(thetaL1), detected_objects[j]->timestamp);
            track.l_shape_filter_ = l_shape_filt;
            track.center_ = detected_objects[j]->bbox.center;
            track.velocity_ = Eigen::Vector3f::Zero();

            track.corners_[0] << detected_objects[j]->bbox.corners2d(0, 0), detected_objects[j]->bbox.corners2d(1, 0),
                0.0f;
            track.corners_[1] << detected_objects[j]->bbox.corners2d(0, 1), detected_objects[j]->bbox.corners2d(1, 1),
                0.0f;
            track.corners_[2] << detected_objects[j]->bbox.corners2d(0, 2), detected_objects[j]->bbox.corners2d(1, 2),
                0.0f;
            track.corners_[3] << detected_objects[j]->bbox.corners2d(0, 3), detected_objects[j]->bbox.corners2d(1, 3),
                0.0f;
            track.type_deq_.emplace_back(detected_objects[j]->type);
            tracklets_.emplace_back(track);
        }
    }
}

void LShapeTrack::CollectTrackingReault(std::vector<common::ObjectPtr>& objects_out) {
    objects_out.clear();

    for (auto& t : tracklets_) {
        // 将跟踪结果放到历史中
        common::ObjectPtr obj_history = std::make_shared<common::Object>();
        obj_history->timestamp = t.last_detect_obj_->timestamp;
        obj_history->velocity = t.velocity_;
        obj_history->type = t.obj_type_;
        obj_history->bbox.center = t.center_;
        obj_history->bbox.size = t.size_;
        t.object_tracked_buffer_.push_back(obj_history);
        if (t.object_tracked_buffer_.size() > 50) {
            t.object_tracked_buffer_.pop_front();
        }

        if (t.life_time_ < 1) {
            continue;
        }

        ///  rule-based 方法均为unknown，也保留
        // if (t.obj_type_ == common::ObjectType::UNKNOWN) {
        //     continue;
        // }

        common::ObjectPtr obj_out = std::make_shared<common::Object>();
        obj_out->track_id = t.track_id_;
        obj_out->type = t.obj_type_;
        obj_out->age = t.life_time_;
        obj_out->velocity = t.velocity_;
        obj_out->state_covariance = t.state_cov_;

        // TLOG_INFO << " track id:" << obj_out->track_id << "  state covariance:\n" << obj_out->state_covariance;

        // Copy boundingbox info
        obj_out->bbox.corners2d.col(0) << t.corners_[0].x(), t.corners_[0].y();
        obj_out->bbox.corners2d.col(1) << t.corners_[1].x(), t.corners_[1].y();
        obj_out->bbox.corners2d.col(2) << t.corners_[2].x(), t.corners_[2].y();
        obj_out->bbox.corners2d.col(3) << t.corners_[3].x(), t.corners_[3].y();
        obj_out->bbox.size = t.size_;
        obj_out->bbox.center = t.center_;
        obj_out->bbox.theta = t.direction_;

        obj_out->lshape_box.reference_point.x() = t.reference_point_.x();
        obj_out->lshape_box.reference_point.y() = t.reference_point_.y();

        // TODO: 缓存的历史目标信息如何传出去
        // temp.obj_history = t.object_tracked_history_;

        /// Trick:
        /// 1、速度限制， < 50m/s
        /// 2、近fov区内速度校正回归
        // 速度 > 50m/s (180km/h) continue
        float velocity_abs = hypotf(obj_out->velocity.x(), obj_out->velocity.y());
        if (velocity_abs > 50.0f) {
            continue;
        }
        // 不在FOV区内，速度矫正
        if (!depthink::perception::lidar::utils::inFOV(obj_out->bbox.center.x(), obj_out->bbox.center.y(), fov_,
                                                    delta_x_correction_velocity_)) {
            if (obj_out->bbox.center.y() > 1.0f && obj_out->bbox.center.y() < 4.5f && obj_out->velocity.y() > -1.5f &&
                obj_out->velocity.y() < 0.0f) {
                obj_out->velocity.y() = 0.0f;
            } else if (obj_out->bbox.center.y() < -1.0f && obj_out->bbox.center.y() > -4.5f &&
                       obj_out->velocity.y() < 1.5f && obj_out->velocity.y() > 0.0f) {
                obj_out->velocity.y() = 0.0f;
            }
        }

        objects_out.emplace_back(obj_out);
    }
}

bool LShapeTrack::Track(common::LidarFramePtr& frame_data) {
    std::vector<common::ObjectPtr>& detected_objects = frame_data->detected_objects;
    const auto& current_time_stamp = frame_data->timestamp;

    /// 将tracker预测到当前时刻
    for (auto& t : tracklets_) {
        t.predict(frame_data->tf, current_time_stamp);
    }

    // TLOG_INFO << "--------------------tracklet size:" << tracklets_.size() << "  detected size:" << detected_objects.size();

    // Associate objects and tracks
    std::vector<int> assignments;
    ObjectsAssociation(detected_objects, current_time_stamp, assignments);

    // update unassigned tracks
    UpdateUnassignedTracks(assignments);

    // Update unassigned objects
    UpdateUnassignedObjects(assignments, detected_objects);

    // update assigned tracks
    UpdateAssignedTracks(detected_objects, current_time_stamp, assignments);

    // Collect tracking result
    CollectTrackingReault(frame_data->tracked_objects);
    return true;
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

