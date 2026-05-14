
#include "motion_filter/multi_anchor/measurement_collection.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

namespace utils {
// @brief calculate the size and center of the bounding-box of a point cloud
// old name: compute_bbox_size_center_xy
template <typename PointCloudT>
void CalculateBBoxSizeCenter2DXY(
    const PointCloudT &cloud, const Eigen::Vector3f &dir, Eigen::Vector3f *size,
    Eigen::Vector3f *center,
    float minimum_edge_length = std::numeric_limits<float>::epsilon()) {
  // NOTE: direction should not be (0, 0, 1)
  Eigen::Matrix3f projection;
  Eigen::Vector3f dird(dir[0], dir[1], 0.0);
  dird.normalize();
  // 绕z轴旋转  (x, y) * |  cos(t)  sin(t) |
  //                    | -sin(t)  cos(t) |    需要验证一下旋转方向的的正负
  projection << dird[0], dird[1], 0.0, -dird[1], dird[0], 0.0, 0.0, 0.0, 1.0;
  constexpr float kFloatMax = std::numeric_limits<float>::max();
  Eigen::Vector3f min_pt(kFloatMax, kFloatMax, kFloatMax);
  Eigen::Vector3f max_pt(-kFloatMax, -kFloatMax, -kFloatMax);
  Eigen::Vector3f loc_pt(0.0, 0.0, 0.0);
  for (size_t i = 0; i < cloud.size(); i++) {  // 旋转到与x，y轴平行后计算其最大最小值即为 size
    loc_pt = projection * Eigen::Vector3f(cloud[i].x, cloud[i].y, cloud[i].z);

    min_pt(0) = std::min(min_pt(0), loc_pt(0));
    min_pt(1) = std::min(min_pt(1), loc_pt(1));
    min_pt(2) = std::min(min_pt(2), loc_pt(2));

    max_pt(0) = std::max(max_pt(0), loc_pt(0));
    max_pt(1) = std::max(max_pt(1), loc_pt(1));
    max_pt(2) = std::max(max_pt(2), loc_pt(2));
  }
  (*size) = (max_pt - min_pt);
  Eigen::Vector3f coeff = (max_pt + min_pt) * 0.5;
  coeff(2) = min_pt(2);
  *center = projection.transpose() * coeff;

  constexpr float kFloatEpsilon = std::numeric_limits<float>::epsilon();
  float minimum_size = std::max(minimum_edge_length, kFloatEpsilon);

  (*size)(0) = (*size)(0) <= minimum_size ? minimum_size : (*size)(0);
  (*size)(1) = (*size)(1) <= minimum_size ? minimum_size : (*size)(1);
  (*size)(2) = (*size)(2) <= minimum_size ? minimum_size : (*size)(2);
}


// @brief calculate the project vector from one vector to another
// old name: compute_2d_xy_project_vector
template <typename T>
Eigen::Matrix<T, 3, 1> Calculate2DXYProjectVector(
    const Eigen::Matrix<T, 3, 1> &projected_vector,
    const Eigen::Matrix<T, 3, 1> &project_vector) {
  if (projected_vector.head(2).norm() < std::numeric_limits<T>::epsilon() ||
      project_vector.head(2).norm() < std::numeric_limits<T>::epsilon()) {
    return Eigen::Matrix<T, 3, 1>::Zero(3, 1);
  }
  Eigen::Matrix<T, 3, 1> project_dir = project_vector;
  project_dir(2) = 0.0;
  project_dir.normalize();

  const T projected_vector_project_dir_inner_product =
      projected_vector(0) * project_dir(0) +
      projected_vector(1) * project_dir(1);
  const T projected_vector_project_dir_angle_cos =
      projected_vector_project_dir_inner_product /
      (projected_vector.head(2).norm() * project_dir.head(2).norm());  // 计算向量夹角的余旋值
  const T projected_vector_norm_on_project_dir =
      projected_vector.head(2).norm() * projected_vector_project_dir_angle_cos;
  
  return project_dir * projected_vector_norm_on_project_dir;
}

} // namespace utils

/////////////////////////////////////////
/// Velocity calculation
/////////////////////////////////////////

// anchor_point的位置之差除以时间
void MeasureAnchorPointVelocity(TrackedObjectPtr new_object,
                                const TrackedObjectConstPtr& old_object,
                                const double& time_diff) {
  // 安全检查
  if (!new_object || !old_object || time_diff < 1e-6) {
    if (new_object) {
      new_object->measured_barycenter_velocity = Eigen::Vector3f::Zero();
    }
    return;
  }

  // Compute 2D anchor point velocity measurement
  Eigen::Vector3f measured_anchor_point_velocity =
      new_object->anchor_point - old_object->anchor_point;
  measured_anchor_point_velocity /= time_diff;
  measured_anchor_point_velocity(2) = 0.0;

  new_object->measured_barycenter_velocity = measured_anchor_point_velocity;
}

// detect bbox center_point的位置之差除以时间
void MeasureDetectBboxCenterPointVelocity(TrackedObjectPtr new_object,
                                const TrackedObjectConstPtr& old_object,
                                const double& time_diff) {
  // 安全检查
  if (!new_object || !old_object || !new_object->object_ptr || time_diff < 1e-6) {
    new_object->measured_detect_bbox_center_velocity = Eigen::Vector3f::Zero();
    return;
  }

  Eigen::Vector3f measured_velocity =
      new_object->object_ptr->bbox.center - old_object->output_center;
  measured_velocity /= time_diff;
  measured_velocity(2) = 0.0;

  new_object->measured_detect_bbox_center_velocity = measured_velocity;
}

void MeasureBboxCenterVelocity(TrackedObjectPtr new_object,
                               const TrackedObjectConstPtr& old_object,
                               const double& time_diff) {
  // 安全检查
  if (!new_object || !old_object || !new_object->object_ptr || 
      !new_object->object_ptr->points || time_diff < 1e-6) {
    new_object->measured_center_velocity = Eigen::Vector3f::Zero();
    return;
  }

  // Compute 2D bbox center velocity measurement
  Eigen::Vector3f old_dir_tmp = old_object->output_direction;
  // Eigen::Vector3f old_size = old_object->output_size;
  Eigen::Vector3f old_center = old_object->output_center;
  Eigen::Vector3f new_size_tmp;
  Eigen::Vector3f new_center;
  float minimum_edge_length = 0.01f;
  common::PointCloud& cloud = *(new_object->object_ptr->points); // 此处需要将点云旋转到当前的全局坐标系下
                                                               // 将上一帧的center，points旋转到当前全局坐标系
  
  // 检查点云是否为空
  if (cloud.empty()) {
    new_object->measured_center_velocity = Eigen::Vector3f::Zero();
    return;
  }
  
  // 根据新输入的点云计算其 center 和 size
  utils::CalculateBBoxSizeCenter2DXY(cloud, old_dir_tmp, &new_size_tmp,
                                      &new_center, minimum_edge_length);
  // Eigen::Vector3f old_dir = old_dir_tmp;
  // Eigen::Vector3f new_size = new_size_tmp;
  Eigen::Vector3f measured_bbox_center_velocity_with_old_dir =
      (new_center - old_center);
  measured_bbox_center_velocity_with_old_dir /= time_diff;
  measured_bbox_center_velocity_with_old_dir(2) = 0.0;
  Eigen::Vector3f measured_bbox_center_velocity =
      measured_bbox_center_velocity_with_old_dir;
  // Eigen::Vector3f project_dir =
  //     new_object->anchor_point - old_object->anchor_point;
  
  // 加了一个判断，当航向角和速度方向小于零时认为通过中心点计算的速度是不准确的
  // if (measured_bbox_center_velocity.dot(project_dir) <= 0) {
  //   measured_bbox_center_velocity.fill(std::numeric_limits<float>::infinity());
  // }
  new_object->measured_center_velocity = measured_bbox_center_velocity;
}

void MeasureBboxCornerVelocity(TrackedObjectPtr new_object,
                               const TrackedObjectConstPtr& old_object,
                               const double& time_diff) {
  // 安全检查
  if (!new_object || !old_object || !new_object->object_ptr || 
      !new_object->object_ptr->points || time_diff < 1e-6) {
    for (int i = 0; i < 4; ++i) {
      new_object->measured_corners_velocity[i] = Eigen::Vector3f::Zero();
    }
    return;
  }

  // Compute 2D bbxo corner velocity measurement
  Eigen::Vector3f old_dir_tmp = old_object->output_direction;
  Eigen::Vector3f old_size = old_object->output_size;
  Eigen::Vector3f old_center = old_object->output_center;
  Eigen::Vector3f new_size_tmp;
  Eigen::Vector3f new_center;
  float minimum_edge_length = 0.01f;
  common::PointCloud& cloud = *(new_object->object_ptr->points);

  // 检查点云是否为空
  if (cloud.empty()) {
    for (int i = 0; i < 4; ++i) {
      new_object->measured_corners_velocity[i] = Eigen::Vector3f::Zero();
    }
    return;
  }

  // 计算新目标的 center 和 size
  utils::CalculateBBoxSizeCenter2DXY(cloud, old_dir_tmp, &new_size_tmp,
                                      &new_center, minimum_edge_length);
  Eigen::Vector3f old_dir = old_dir_tmp;
  Eigen::Vector3f new_size = new_size_tmp;
  Eigen::Vector3f ortho_old_dir(-old_dir(1), old_dir(0), 0.0);

  // 算4个corner点的位置
  Eigen::Vector3f old_bbox_corner_list[4];
  Eigen::Vector3f new_bbox_corner_list[4];
  Eigen::Vector3f old_bbox_corner = old_center + old_dir * old_size(0) * 0.5 +
                                    ortho_old_dir * old_size(1) * 0.5;
  Eigen::Vector3f new_bbox_corner = new_center + old_dir * new_size(0) * 0.5 +
                                    ortho_old_dir * new_size(1) * 0.5;
  old_bbox_corner_list[0] = old_bbox_corner;  // 左前方
  new_bbox_corner_list[0] = new_bbox_corner;
  old_bbox_corner = old_center - old_dir * old_size(0) * 0.5 +
                    ortho_old_dir * old_size(1) * 0.5;
  new_bbox_corner = new_center - old_dir * new_size(0) * 0.5 +
                    ortho_old_dir * new_size(1) * 0.5;
  old_bbox_corner_list[1] = old_bbox_corner;  // 左后方
  new_bbox_corner_list[1] = new_bbox_corner;
  old_bbox_corner = old_center + old_dir * old_size(0) * 0.5 -
                    ortho_old_dir * old_size(1) * 0.5;
  new_bbox_corner = new_center + old_dir * new_size(0) * 0.5 -
                    ortho_old_dir * new_size(1) * 0.5;
  old_bbox_corner_list[2] = old_bbox_corner;  // 右前方
  new_bbox_corner_list[2] = new_bbox_corner;
  old_bbox_corner = old_center - old_dir * old_size(0) * 0.5 -
                    ortho_old_dir * old_size(1) * 0.5;
  new_bbox_corner = new_center - old_dir * new_size(0) * 0.5 -
                    ortho_old_dir * new_size(1) * 0.5;
  old_bbox_corner_list[3] = old_bbox_corner;  // 右后方
  new_bbox_corner_list[3] = new_bbox_corner;

  // calculate the nearest corner  在四个角点中找一个离本车最近的corner点
  // Eigen::Vector3f ref_location = new_object->sensor_to_local_pose.translation();  // TODO: 位置需要确认
  Eigen::Vector3f ref_location = Eigen::Vector3f::Zero();
  Eigen::Vector3f nearest_old_bbox_corner = old_bbox_corner_list[0];
  Eigen::Vector3f nearest_new_bbox_corner = new_bbox_corner_list[0];
  double min_center_distance = (ref_location - nearest_new_bbox_corner).norm();
  for (size_t i = 1; i < 4; ++i) {
    double center_distance = (ref_location - new_bbox_corner_list[i]).norm();
    if (center_distance < min_center_distance) {
      min_center_distance = center_distance;
      nearest_old_bbox_corner = old_bbox_corner_list[i];
      nearest_new_bbox_corner = new_bbox_corner_list[i];
    }
  }
  // no projection  直接用最近的角点距离差除以时间
  new_object->measured_nearest_corner_velocity =
      (nearest_new_bbox_corner - nearest_old_bbox_corner) / time_diff;

  // select project_dir by change of size
  Eigen::Vector3f direct_old_size = old_object->size;
  Eigen::Vector3f direct_new_size = new_object->size;  // 需要确定这个新的size是何时在哪里计算的
                                                       // 猜测来自检测的结果？ 可以在 tracked_object.attach()中看看
  double length_change =
      fabs(direct_old_size(0) - direct_new_size(0)) / direct_old_size(0);  // 计算size改变的比例大小
  double width_change =
      fabs(direct_old_size(1) - direct_new_size(1)) / direct_old_size(1);

  const double max_change_thresh = 0.1;
  Eigen::Vector3f project_dir;
  // 如果size改变太大，则不使用bbox的中心点作为投影向量， 使用anchor_point的变化作为投影向量
  if (length_change < max_change_thresh && width_change < max_change_thresh) {
    project_dir = new_object->center - old_object->center;
  } else {
    project_dir = new_object->anchor_point - old_object->anchor_point;
  }

  for (size_t i = 0; i < 4; ++i) {
    old_bbox_corner = old_bbox_corner_list[i];
    new_bbox_corner = new_bbox_corner_list[i];
    new_object->corners[i] = new_bbox_corner_list[i];
    Eigen::Vector3f bbox_corner_velocity =
        ((new_bbox_corner - old_bbox_corner) / time_diff);
    Eigen::Vector3f bbox_corner_velocity_on_project_dir =
        utils::Calculate2DXYProjectVector<float>(bbox_corner_velocity,  // bbox corner的速度往 project_dir 方向上的投影长度
                                                   project_dir);
    // set velocity as 0 when conflict  速度差异超过90度时认为不可靠，直接置零
    if (bbox_corner_velocity_on_project_dir.dot(project_dir) <= 0) {
      bbox_corner_velocity_on_project_dir = Eigen::Vector3f::Zero();
    }
    new_object->measured_corners_velocity[i] =
        bbox_corner_velocity_on_project_dir;
  }
}

// 检测bbox角点速度
void MeasureDetectBboxCornerVelocity(TrackedObjectPtr new_object,
                               const TrackedObjectConstPtr& old_object,
                               const double& time_diff) {
  // 安全检查
  if (!new_object || !old_object || !new_object->object_ptr || time_diff < 1e-6) {
    for (int i = 0; i < 4; ++i) {
      new_object->measured_corners_velocity[i] = Eigen::Vector3f::Zero();
    }
    return;
  }

  // Compute 2D bbxo corner velocity measurement
  Eigen::Vector3f old_dir_tmp = old_object->output_direction;
  Eigen::Vector3f old_size = old_object->output_size;
  Eigen::Vector3f old_center = old_object->output_center;
  Eigen::Vector3f new_size_tmp = new_object->object_ptr->bbox.size;
  Eigen::Vector3f new_center = new_object->object_ptr->bbox.center;
  // float minimum_edge_length = 0.01f;
  // common::PointCloud& cloud = *(new_object->object_ptr->points);

  // // 计算新目标的 center 和 size
  // utils::CalculateBBoxSizeCenter2DXY(cloud, old_dir_tmp, &new_size_tmp,
  //                                     &new_center, minimum_edge_length);
  Eigen::Vector3f old_dir = old_dir_tmp;
  Eigen::Vector3f new_size = new_size_tmp;
  Eigen::Vector3f ortho_old_dir(-old_dir(1), old_dir(0), 0.0);

  // 算4个corner点的位置
  Eigen::Vector3f old_bbox_corner_list[4];
  Eigen::Vector3f new_bbox_corner_list[4];
  Eigen::Vector3f old_bbox_corner = old_center + old_dir * old_size(0) * 0.5 +
                                    ortho_old_dir * old_size(1) * 0.5;
  Eigen::Vector3f new_bbox_corner = new_center + old_dir * new_size(0) * 0.5 +
                                    ortho_old_dir * new_size(1) * 0.5;
  old_bbox_corner_list[0] = old_bbox_corner;  // 左前方
  new_bbox_corner_list[0] = new_bbox_corner;
  old_bbox_corner = old_center - old_dir * old_size(0) * 0.5 +
                    ortho_old_dir * old_size(1) * 0.5;
  new_bbox_corner = new_center - old_dir * new_size(0) * 0.5 +
                    ortho_old_dir * new_size(1) * 0.5;
  old_bbox_corner_list[1] = old_bbox_corner;  // 左后方
  new_bbox_corner_list[1] = new_bbox_corner;
  old_bbox_corner = old_center + old_dir * old_size(0) * 0.5 -
                    ortho_old_dir * old_size(1) * 0.5;
  new_bbox_corner = new_center + old_dir * new_size(0) * 0.5 -
                    ortho_old_dir * new_size(1) * 0.5;
  old_bbox_corner_list[2] = old_bbox_corner;  // 右前方
  new_bbox_corner_list[2] = new_bbox_corner;
  old_bbox_corner = old_center - old_dir * old_size(0) * 0.5 -
                    ortho_old_dir * old_size(1) * 0.5;
  new_bbox_corner = new_center - old_dir * new_size(0) * 0.5 -
                    ortho_old_dir * new_size(1) * 0.5;
  old_bbox_corner_list[3] = old_bbox_corner;  // 右后方
  new_bbox_corner_list[3] = new_bbox_corner;

  // calculate the nearest corner  在四个角点中找一个离本车最近的corner点
  // Eigen::Vector3f ref_location = new_object->sensor_to_local_pose.translation();  // TODO: 位置需要确认
  Eigen::Vector3f ref_location = Eigen::Vector3f::Zero();
  Eigen::Vector3f nearest_old_bbox_corner = old_bbox_corner_list[0];
  Eigen::Vector3f nearest_new_bbox_corner = new_bbox_corner_list[0];
  double min_center_distance = (ref_location - nearest_new_bbox_corner).norm();
  for (size_t i = 1; i < 4; ++i) {
    double center_distance = (ref_location - new_bbox_corner_list[i]).norm();
    if (center_distance < min_center_distance) {
      min_center_distance = center_distance;
      nearest_old_bbox_corner = old_bbox_corner_list[i];
      nearest_new_bbox_corner = new_bbox_corner_list[i];
    }
  }
  // no projection  直接用最近的角点距离差除以时间
  // new_object->measured_nearest_corner_velocity =
  //     (nearest_new_bbox_corner - nearest_old_bbox_corner) / time_diff;

  // select project_dir by change of size
  Eigen::Vector3f direct_old_size = old_object->size;
  Eigen::Vector3f direct_new_size = new_object->size;  // 需要确定这个新的size是何时在哪里计算的
                                                       // 猜测来自检测的结果？ 可以在 tracked_object.attach()中看看
  double length_change =
      fabs(direct_old_size(0) - direct_new_size(0)) / direct_old_size(0);  // 计算size改变的比例大小
  double width_change =
      fabs(direct_old_size(1) - direct_new_size(1)) / direct_old_size(1);

  const double max_change_thresh = 0.1;
  Eigen::Vector3f project_dir;
  // 如果size改变太大，则不使用bbox的中心点作为投影向量， 使用anchor_point的变化作为投影向量
  if (length_change < max_change_thresh && width_change < max_change_thresh) {
    project_dir = new_center - old_center;
  } else {
    project_dir = new_object->anchor_point - old_object->anchor_point;
  }

  for (size_t i = 0; i < 4; ++i) {
    old_bbox_corner = old_bbox_corner_list[i];
    new_bbox_corner = new_bbox_corner_list[i];
    new_object->corners[i] = new_bbox_corner_list[i];
    Eigen::Vector3f bbox_corner_velocity =
        ((new_bbox_corner - old_bbox_corner) / time_diff);

    Eigen::Vector3f bbox_corner_velocity_on_project_dir =
        utils::Calculate2DXYProjectVector<float>(bbox_corner_velocity,  // bbox corner的速度往 project_dir 方向上的投影长度
                                                   project_dir);
    // set velocity as 0 when conflict  速度差异超过90度时认为不可靠，直接置零
    // if (bbox_corner_velocity_on_project_dir.dot(project_dir) <= 0) {
    //   bbox_corner_velocity_on_project_dir = Eigen::Vector3f::Zero();
    //   bbox_corner_velocity_on_project_dir.fill(std::numeric_limits<float>::infinity());
    // }
    new_object->measured_corners_velocity[i] =
        bbox_corner_velocity_on_project_dir;
  }

}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
