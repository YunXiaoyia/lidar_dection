#include "obj_tracked_3d.h"

// #include "common_utils.h"


#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {



Eigen::Vector3f RotatePoint(const Eigen::Vector3f& p1, const Eigen::Vector3f& base_point, const double& theta) {
    if (theta > M_PI / 2.0) {
        return p1;
    }

    Eigen::Vector3f res = Eigen::Vector3f::Zero();

    res[0] = (p1.x() - base_point.x()) * cos(theta) - (p1.y() - base_point.y()) * sin(theta) + base_point.x();
    res[1] = (p1.y() - base_point.y()) * cos(theta) + (p1.x() - base_point.x()) * sin(theta) + base_point.y();

    return res;
}


void bboxByDetect(const common::ObjectPtr& detect_obj, Eigen::Vector3f& center, Eigen::Vector3f& size,
                  std::array<Eigen::Vector3f, 4>& corner_point_array, float& direction_out) {
    center = detect_obj->bbox.center;
    size = detect_obj->bbox.size;

    corner_point_array[0] << detect_obj->bbox.corners2d(0, 0), detect_obj->bbox.corners2d(1, 0), 0.0f;
    corner_point_array[1] << detect_obj->bbox.corners2d(0, 1), detect_obj->bbox.corners2d(1, 1), 0.0f;
    corner_point_array[2] << detect_obj->bbox.corners2d(0, 2), detect_obj->bbox.corners2d(1, 2), 0.0f;
    corner_point_array[3] << detect_obj->bbox.corners2d(0, 3), detect_obj->bbox.corners2d(1, 3), 0.0f;

    direction_out = detect_obj->bbox.theta;
}


void correctBboxByVelocity(const Eigen::Vector3f& velocity, std::array<Eigen::Vector3f, 4>& corner_point_array,
                           float& direction_out, Eigen::Vector3f& center_point_out) {
    static const double LAMBDA = 0.9;

    // 速度太小（<2.0m/s），则不使用速度进行校正
    if (velocity.norm() < 0.1f || velocity.x() < -10.0) {
        return;
    }

    double velocity_direction = atan2(velocity.y(), velocity.x());

    //     c1---c2
    //      |    |
    //      |    |
    //     c3---c4
    Eigen::Vector3f c1, c2, c3, c4;
    std::sort(corner_point_array.begin(), corner_point_array.end(),
              [](const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) { return p1.x() < p2.x(); });
    if (corner_point_array[0].y() > corner_point_array[1].y()) {
        c3 = corner_point_array[0];
        c4 = corner_point_array[1];
    } else {
        c3 = corner_point_array[1];
        c4 = corner_point_array[0];
    }

    if (corner_point_array[2].y() > corner_point_array[3].y()) {
        c1 = corner_point_array[2];
        c2 = corner_point_array[3];
    } else {
        c1 = corner_point_array[3];
        c2 = corner_point_array[2];
    }

    Eigen::Vector3f base_point = (c3 + c4) / 2.0;

    double direction1 = atan2(c1.y() - c3.y(), c1.x() - c3.x());
    double direction2 = atan2(c4.y() - c3.y(), c4.x() - c3.x());

    if (fabs(velocity_direction - direction1) < fabs(velocity_direction - direction2)) {
        double rotate_angle = LAMBDA * (velocity_direction - direction1);

        corner_point_array[0] = RotatePoint(c1, base_point, rotate_angle);
        corner_point_array[1] = RotatePoint(c2, base_point, rotate_angle);
        corner_point_array[3] = RotatePoint(c3, base_point, rotate_angle);
        corner_point_array[2] = RotatePoint(c4, base_point, rotate_angle);

        // direction_out = atan2(c1.y() - c3.y(), c1.x() - c3.x());
        direction_out = atan2(corner_point_array[0].y() - corner_point_array[3].y(),
                              corner_point_array[0].x() - corner_point_array[3].x());

        center_point_out = RotatePoint(center_point_out, base_point, rotate_angle);

    } else {
        double rotate_angle = LAMBDA * (velocity_direction - direction2);

        corner_point_array[0] = RotatePoint(c1, base_point, rotate_angle);
        corner_point_array[1] = RotatePoint(c2, base_point, rotate_angle);
        corner_point_array[3] = RotatePoint(c3, base_point, rotate_angle);
        corner_point_array[2] = RotatePoint(c4, base_point, rotate_angle);

        direction_out = atan2(corner_point_array[2].y() - corner_point_array[3].y(),
                              corner_point_array[2].x() - corner_point_array[3].x());
        center_point_out = RotatePoint(center_point_out, base_point, rotate_angle);
    }
}


ObjTracked3d::ObjTracked3d() {}

ObjTracked3d::ObjTracked3d(const ObjTracked3d& track) {
    // this->times_tamp = track.times_tamp;
    this->track_id_ = track.track_id_;
    this->life_time_ = track.life_time_;
    this->consecutive_lost_ = track.consecutive_lost_;
    this->type_deq_ = track.type_deq_;
    this->obj_type_ = track.obj_type_;

    this->last_detect_obj_ = track.last_detect_obj_;
    this->l_shape_filter_ = track.l_shape_filter_;
    this->center_ = track.center_;
    this->velocity_ = track.velocity_;
    this->corners_ = track.corners_;
    this->size_ = track.size_;
    this->direction_ = track.direction_;

    this->object_tracked_buffer_ = track.object_tracked_buffer_;

    this->object_detected_buffer_ = track.object_detected_buffer_;
}

ObjTracked3d& ObjTracked3d::operator=(const ObjTracked3d& track) {
    if (this == &track) {
        return *this;
    }

    // this->times_tamp = track.times_tamp;
    this->track_id_ = track.track_id_;
    this->life_time_ = track.life_time_;
    this->consecutive_lost_ = track.consecutive_lost_;
    this->type_deq_ = track.type_deq_;
    this->obj_type_ = track.obj_type_;

    this->last_detect_obj_ = track.last_detect_obj_;
    this->l_shape_filter_ = track.l_shape_filter_;
    this->center_ = track.center_;
    this->velocity_ = track.velocity_;
    this->corners_ = track.corners_;
    this->size_ = track.size_;
    this->direction_ = track.direction_;

    this->object_tracked_buffer_ = track.object_tracked_buffer_;

    this->object_detected_buffer_ = track.object_detected_buffer_;

    return *this;
}

void ObjTracked3d::predict(const Eigen::Isometry3f& tf, const double& stamp) {
    double dt = stamp - last_detect_obj_->timestamp;
    if (consecutive_lost_ > 0) {
        center_ = tf * (center_ + velocity_ * 0.099);
        for (auto& p : corners_) {
            p = tf * (p + velocity_ * 0.099);
        }
    } else {
        center_ = tf * (center_ + velocity_ * dt);
        for (auto& p : corners_) {
            p = tf * (p + velocity_ * dt);
        }
    }

    velocity_ = tf.rotation() * velocity_;
    l_shape_filter_.transformToCurrent(tf.cast<double>());
    l_shape_filter_.predict(stamp);
    reference_point_ = l_shape_filter_.getMotionState().head(2);
    state_cov_ = l_shape_filter_.dynamic_kf.stateCovariance().block<4, 4>(0, 0).cast<float>();

    // 目标历史帧数据的转换
    for (auto& tracked : object_tracked_buffer_) {
        auto& center_history = tracked->bbox.center;
        center_history = tf * center_history;
    }
}

void normalizeAngleByVelocity(const Eigen::Vector3f& velocity, Eigen::Vector3f& size, float& direction) {
    Eigen::Vector3f size_tmp = size;
    double angle_norm = normalize_angle(direction);

    std::vector<float> angles;
    angles.push_back(angle_norm);
    angles.push_back(angle_norm + M_PI);
    angles.push_back(angle_norm + M_PI / 2);
    angles.push_back(angle_norm + 3 * M_PI / 2);

    double vsp = atan2(velocity.y(), velocity.x());

    double min = 1.56;
    double distance = 0.0, orientation = 0.0;
    int pos = -1;
    for (int i = 0; i < 4; ++i) {
        distance = abs(shortest_angular_distance(vsp, angles[i]));
        if (distance < min) {
            min = distance;
            orientation = normalize_angle(angles[i]);
            pos = i;
        }
    }

    if (pos == 0 || pos == 1) {
        size.x() = size_tmp.x();
        size.y() = size_tmp.y();
    } else {
        size.x() = size_tmp.y();
        size.y() = size_tmp.x();
    }

    direction = normalize_angle(orientation);
}

void ObjTracked3d::update(const common::ObjectPtr& detect, const double& cur_time) {
    // LShape特征
    double x_corner = detect->lshape_box.reference_point(0);
    double y_corner = detect->lshape_box.reference_point(1);
    double L1 = detect->lshape_box.l_shape(0);
    double L2 = detect->lshape_box.l_shape(1);
    double thetaL1 = detect->lshape_box.l_shape(2);

    last_detect_obj_ = detect;
    // 更新
    l_shape_filter_.update(x_corner, y_corner, L1, L2, thetaL1, cur_time);
    // 取出状态
    l_shape_filter_.BoxModel(center_, velocity_, corners_, size_, direction_);
    // 使用速度对Bbox方向进行修正;
    // if (life_time_ > 5 && center_.x() > 30.0f && velocity_.norm() > 1.0f && velocity_.x() > -10.0f) {
    //     correctBboxByVelocity(velocity_, corners_, direction_, center_);
    // }

    // else

    bboxByDetect(last_detect_obj_, center_, size_, corners_, direction_);
    normalizeAngleByVelocity(velocity_, size_, direction_);

    // center(2) = (proposal.min_z + proposal.max_z) / 2;
    center_(2) = last_detect_obj_->bbox.center(2);
    // extent(2) = std::max((proposal.max_z - proposal.min_z), 0.5f);
    size_(2) = last_detect_obj_->bbox.size(2);


    /// 取出状态点
    reference_point_ = l_shape_filter_.getMotionState().head(2);

    /// 取出状态协方差
    state_cov_ = l_shape_filter_.dynamic_kf.stateCovariance().block<4, 4>(0, 0).cast<float>();

    // life_time_ += 1;
    life_time_ = std::min(life_time_, int16_t(999)) + 1;
    consecutive_lost_ = 0;

    type_deq_.emplace_back(detect->type);

    /// 类型投票
    if(type_deq_.size() >= 3){
        std::vector<int> obj_vote(int(common::ObjectType::TYPE_NUM));
        for(auto& type : type_deq_){
            obj_vote.at((int)type)++;
        }
        int type = std::max_element(obj_vote.begin(), obj_vote.end()) - obj_vote.begin();
        obj_type_ = common::ObjectType(type);
    }

    if (type_deq_.size() >= 29) {
        type_deq_.pop_front();
    }

    // Cache detected object data
    object_detected_buffer_.push_back(detect);
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
