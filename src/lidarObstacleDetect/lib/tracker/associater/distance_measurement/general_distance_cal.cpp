
#include "associater/distance_measurement/general_distance_cal.h"

#include "common/Macros.h"

HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

namespace utils {

template <typename T>
T CalculateCosTheta2DXY(const Eigen::Matrix<T, 3, 1> &v1,
                        const Eigen::Matrix<T, 3, 1> &v2) {
  T v1_len = static_cast<T>(sqrt((v1.head(2).cwiseProduct(v1.head(2))).sum()));
  T v2_len = static_cast<T>(sqrt((v2.head(2).cwiseProduct(v2.head(2))).sum()));
  if (v1_len < std::numeric_limits<T>::epsilon() ||
      v2_len < std::numeric_limits<T>::epsilon()) {
    return 0.0;
  }
  T cos_theta = (v1.head(2).cwiseProduct(v2.head(2))).sum() / (v1_len * v2_len);
  return cos_theta;
}

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
  constexpr double kDoubleMax = std::numeric_limits<double>::max();
  Eigen::Vector3f min_pt(kDoubleMax, kDoubleMax, kDoubleMax);
  Eigen::Vector3f max_pt(-kDoubleMax, -kDoubleMax, -kDoubleMax);
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
  (*size) = (max_pt - min_pt).cast<float>();
  Eigen::Vector3f coeff = (max_pt + min_pt) * 0.5;
  coeff(2) = min_pt(2);
  *center = projection.transpose() * coeff;

  constexpr float kFloatEpsilon = std::numeric_limits<float>::epsilon();
  float minimum_size = std::max(minimum_edge_length, kFloatEpsilon);

  (*size)(0) = (*size)(0) <= minimum_size ? minimum_size : (*size)(0);
  (*size)(1) = (*size)(1) <= minimum_size ? minimum_size : (*size)(1);
  (*size)(2) = (*size)(2) <= minimum_size ? minimum_size : (*size)(2);
}

// @brief calculate the IOU (intersection-over-union) between two bbox
// old name:compute_2d_iou_bbox_to_bbox
template <typename Type>
Type CalculateIou2DXY(const Eigen::Matrix<Type, 3, 1> &center0,
                      const Eigen::Matrix<Type, 3, 1> &size0,
                      const Eigen::Matrix<Type, 3, 1> &center1,
                      const Eigen::Matrix<Type, 3, 1> &size1) {
  Type min_x_bbox_0 = center0(0) - size0(0) * static_cast<Type>(0.5);
  Type min_x_bbox_1 = center1(0) - size1(0) * static_cast<Type>(0.5);
  Type max_x_bbox_0 = center0(0) + size0(0) * static_cast<Type>(0.5);
  Type max_x_bbox_1 = center1(0) + size1(0) * static_cast<Type>(0.5);
  Type start_x = std::max(min_x_bbox_0, min_x_bbox_1);
  Type end_x = std::min(max_x_bbox_0, max_x_bbox_1);
  Type length_x = end_x - start_x;
  if (length_x <= 0) {
    return 0;
  }
  Type min_y_bbox_0 = center0(1) - size0(1) * static_cast<Type>(0.5);
  Type min_y_bbox_1 = center1(1) - size1(1) * static_cast<Type>(0.5);
  Type max_y_bbox_0 = center0(1) + size0(1) * static_cast<Type>(0.5);
  Type max_y_bbox_1 = center1(1) + size1(1) * static_cast<Type>(0.5);
  Type start_y = std::max(min_y_bbox_0, min_y_bbox_1);
  Type end_y = std::min(max_y_bbox_0, max_y_bbox_1);
  Type length_y = end_y - start_y;
  if (length_y <= 0) {
    return 0;
  }
  Type intersection_area = length_x * length_y;
  Type bbox_0_area = size0(0) * size0(1);
  Type bbox_1_area = size1(0) * size1(1);
  Type iou =
      intersection_area / (bbox_0_area + bbox_1_area - intersection_area);
  return iou;
}

} // namespace utils

// Eigen::Vector3f last_selected_point = latest_object->output_selected_track_point;
  // last_selected_point[0] = last_selected_point[0] + latest_object->output_velocity.x() * time_diff;
  // last_selected_point[1] = last_selected_point[1] + latest_object->output_velocity.y() * time_diff;

  // auto points_eigen = object->points->getMatrixXfMap(3, 8, 0);
  // Eigen::Vector3f anchor_point = points_eigen.rowwise().mean();

float LocationDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff) {
  Eigen::Vector3f measured_center = object->bbox.center;
  Eigen::Vector3f predicted_center = last_object->output_center;
  predicted_center[0] += last_object->output_velocity.x() * time_diff;
  predicted_center[1] += last_object->output_velocity.y() * time_diff;

  Eigen::Vector3f center_diff = measured_center - predicted_center;
  float location_dist = center_diff.head(2).norm(); // 位置之差的模长

  /* NEED TO NOTICE: All the states would be collected mainly based on
   * states of tracked objects. Thus, update tracked object when you
   * update the state of track !!!!! */
  Eigen::Vector2f ref_dir = last_object->output_velocity.head(2);
  double speed = ref_dir.norm();

  /* Let location distance generated from a normal distribution with
   * symmetric variance. Modify its variance when speed greater than
   * a threshold. Penalize variance other than motion direction. */
  if (speed > 2) {
    ref_dir /= speed; // 速度的方向向量

    Eigen::Vector2f ref_o_dir = Eigen::Vector2f(ref_dir[1], -ref_dir[0]);
    double dx = ref_dir(0) * center_diff(0) +
                ref_dir(1) * center_diff(1);
    double dy = ref_o_dir(0) * center_diff(0) +
                ref_o_dir(1) * center_diff(1);
    location_dist = static_cast<float>(sqrt(dx * dx * 0.5 + dy * dy * 2));
  }

  // Avoid excessive lateral acc speed
  double vy = (measured_center.y() - last_object->output_center.y()) / time_diff;
  if (fabs((vy - last_object->output_velocity.y()) / time_diff) > 20) {
    location_dist = location_dist * 2.0;
  }

  return location_dist;
}

float DirectionDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff) {
  // Compute direction distance for last object and new object
  // range from 0 to 2

  Eigen::Vector3f old_anchor_point = last_object->anchor_point;

  auto points_eigen = object->points->getMatrixXfMap(3, 8, 0);
  Eigen::Vector3f new_anchor_point = points_eigen.rowwise().mean();

  Eigen::Vector3f anchor_point_shift_dir = new_anchor_point - old_anchor_point;
  anchor_point_shift_dir[2] = 0.0;

  Eigen::Vector3f track_motion_dir = last_object->output_velocity;
  track_motion_dir[2] = 0.0;

  double cos_theta = 0.994;  // average cos
  if (!track_motion_dir.head(2).isZero() &&
      !anchor_point_shift_dir.head(2).isZero()) {
    // cos_theta = vector_cos_theta_2d_xy(track_motion_dir,
    //                                   anchor_point_shift_dir);
    cos_theta = utils::CalculateCosTheta2DXY<float>(track_motion_dir,
                                                     anchor_point_shift_dir);
  }
  float direction_dist = static_cast<float>(-cos_theta) + 1.0f;

  return direction_dist;
}

float BboxSizeDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff) {
  // Compute bbox size distance for last object and new object
  // range from 0 to 1

  Eigen::Vector3f old_bbox_dir = last_object->output_direction;
  Eigen::Vector3f new_bbox_dir = Eigen::Vector3f(cos(object->bbox.theta), sin(object->bbox.theta), 0.0f);
  Eigen::Vector3f old_bbox_size = last_object->output_size;
  Eigen::Vector3f new_bbox_size = object->bbox.size;

  float size_dist = 0.0f;
  double dot_val_00 = fabs(old_bbox_dir(0) * new_bbox_dir(0) +
                           old_bbox_dir(1) * new_bbox_dir(1));
  double dot_val_01 = fabs(old_bbox_dir(0) * new_bbox_dir(1) -
                           old_bbox_dir(1) * new_bbox_dir(0));
  float temp_val_0 = 0.0f;
  float temp_val_1 = 0.0f;
  if (dot_val_00 > dot_val_01) {
    temp_val_0 = static_cast<float>(fabs(old_bbox_size(0) - new_bbox_size(0))) /
                 std::max(old_bbox_size(0), new_bbox_size(0));
    temp_val_1 = static_cast<float>(fabs(old_bbox_size(1) - new_bbox_size(1))) /
                 std::max(old_bbox_size(1), new_bbox_size(1));
    size_dist = std::min(temp_val_0, temp_val_1);
  } else {
    temp_val_0 = static_cast<float>(fabs(old_bbox_size(0) - new_bbox_size(1))) /
                 std::max(old_bbox_size(0), new_bbox_size(1));
    temp_val_1 = static_cast<float>(fabs(old_bbox_size(1) - new_bbox_size(0))) /
                 std::max(old_bbox_size(1), new_bbox_size(0));
    size_dist = std::min(temp_val_0, temp_val_1);
  }

  return size_dist;
}

// 点数差值与最多点数的比值
float PointNumDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff) {
  // Compute point num distance for last object and new object
  // range from 0 and 2

  int old_point_number = last_object->object_ptr->points->size();
  int new_point_number = object->points->size();

  float point_num_dist =
      static_cast<float>(std::fabs(old_point_number - new_point_number)) * 1.0f /
      static_cast<float>(std::max(old_point_number, new_point_number));

  return point_num_dist;
}

float HistogramDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff) {
  // // Compute histogram distance for last object and new object
  // // range from 0 to 3

  // const std::vector<float>& old_object_shape_features =
  //     last_object->shape_features;
  // const std::vector<float>& new_object_shape_features =
  //     new_object->shape_features;

  // if (old_object_shape_features.size() != new_object_shape_features.size()) {
  //   AINFO << "sizes of compared features not matched. TrackObjectDistance";
  //   return 100;
  // }

  // float histogram_dist = 0.0f;
  // for (size_t i = 0; i < old_object_shape_features.size(); ++i) {
  //   histogram_dist +=
  //       std::fabs(old_object_shape_features[i] - new_object_shape_features[i]);
  // }

  // return histogram_dist;

  return 0;
}

float CentroidShiftDistance(const TrackedObjectConstPtr& last_object,
                            const common::ObjectPtr& object,
                            const double time_diff) {
  const auto& measured_points = object->points->getMatrixXfMap(3, 8, 0);
  const auto& predicted_points = last_object->object_ptr->points->getMatrixXfMap(3, 8, 0);

  const Eigen::Vector3f measured_centroid = measured_points.rowwise().mean();
  Eigen::Vector3f predicted_centroid = predicted_points.rowwise().mean();
  predicted_centroid[0] += last_object->output_velocity.x() * time_diff;
  predicted_centroid[1] += last_object->output_velocity.y() * time_diff;

  const auto centroid_diff = measured_centroid - predicted_centroid;

  float dist = centroid_diff.head(2).norm();
  return dist;
}

float BboxIouDistance(const TrackedObjectConstPtr& last_object,
                      const common::ObjectPtr& object,
                      const double time_diff, float match_threshold) {

  // Step 1: unify bbox direction, change the one with less pts,
  // for efficiency.
  Eigen::Vector3f old_dir = last_object->output_direction;
  Eigen::Vector3f old_size = last_object->output_size;
  Eigen::Vector3f new_dir = Eigen::Vector3f(cos(object->bbox.theta), sin(object->bbox.theta), 0.0f);
  Eigen::Vector3f new_size = object->bbox.size;
  Eigen::Vector3f new_center = object->bbox.center;
  old_dir.normalize();
  new_dir.normalize();
  // handle randomness
  old_size(0) = old_size(0) > 0.3f ? old_size(0) : 0.3f;
  old_size(1) = old_size(1) > 0.3f ? old_size(1) : 0.3f;
  new_size(0) = new_size(0) > 0.3f ? new_size(0) : 0.3f;
  new_size(1) = new_size(1) > 0.3f ? new_size(1) : 0.3f;
  int last_object_num_pts = last_object->object_ptr->points->size();
  int cur_obj_num_pts =object->points->size();

  bool change_cur_obj_bbox = last_object_num_pts > cur_obj_num_pts;
  float minimum_edge_length = 0.01f;

  Eigen::Vector3f predicted_center = last_object->output_center;
  predicted_center[0] = predicted_center[0] + last_object->output_velocity.x() * time_diff;
  predicted_center[1] = predicted_center[1] + last_object->output_velocity.y() * time_diff;

  common::PointCloud& cloud = *(object->points);
  if (change_cur_obj_bbox) {
    new_dir = old_dir;
    utils::CalculateBBoxSizeCenter2DXY(cloud, new_dir, &new_size, &new_center,
                                        minimum_edge_length);
  } else {
    old_dir = new_dir;
    utils::CalculateBBoxSizeCenter2DXY(cloud, old_dir, &old_size, &predicted_center,
                                        minimum_edge_length);
  }
  // Step 2: compute 2d iou bbox to bbox
  Eigen::Matrix2f trans_mat;
  trans_mat(0, 0) = old_dir(0);
  trans_mat(0, 1) = old_dir(1);
  trans_mat(1, 0) = -old_dir(1);
  trans_mat(1, 1) = old_dir(0);
  Eigen::Vector2f old_center_transformed_2d =
      static_cast<Eigen::Matrix<float, 2, 1, 0, 2, 1>>(trans_mat *
                                                        predicted_center.head(2));
  Eigen::Vector2f new_center_transformed_2d =
      static_cast<Eigen::Matrix<float, 2, 1, 0, 2, 1>>(trans_mat *
                                                        new_center.head(2));
  predicted_center(0) = old_center_transformed_2d(0);
  predicted_center(1) = old_center_transformed_2d(1);
  new_center(0) = new_center_transformed_2d(0);
  new_center(1) = new_center_transformed_2d(1);
  Eigen::Vector3f old_size_tmp = old_size;
  Eigen::Vector3f new_size_tmp = new_size;
  float iou = utils::CalculateIou2DXY<float>(predicted_center, old_size_tmp,
                                                new_center, new_size_tmp);
  // Step 4: compute dist
  float dist = (1 - iou) * match_threshold;
  return dist;
}


float BboxTailDistance(const TrackedObjectConstPtr& last_object,
                       const common::ObjectPtr& object,
                       const double time_diff){
  Eigen::Vector3f measured_tail = calTailMiddlePoint(object->bbox);
  Eigen::Vector3f predicted_tail = Eigen::Vector3f(0, 0, 0);

  std::vector<Eigen::Vector3f> last_corners;
  for (int i = 0; i < 4; ++i){
        last_corners.emplace_back(last_object->output_corners[i]);
  }
  std::sort(last_corners.begin(), last_corners.end(), [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b){
        return (a.x() < b.x());
  });
  predicted_tail = (last_corners[0] + last_corners[1]).array() / 2;
  predicted_tail[0] += last_object->output_velocity[0] * time_diff;
  predicted_tail[1] += last_object->output_velocity[1] * time_diff;

  // Todo; 如果tail处于fov边界，return 0
  if(predicted_tail[0] < 15.0f || measured_tail[0] < 15.0f){
      return 0.0;
  }

  const auto tail_diff = measured_tail - predicted_tail;
  float dist = tail_diff.head(2).norm();
   return dist;
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
