
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include "motion_filter/multi_anchor/measurement_collection.h"
#include "motion_filter/multi_anchor/multi_anchor_motion_measurement.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

void MultiAnchorMotionMeasurement::ComputeMotionMeasurment(
    const TrackedObjectConstPtr& latest_object, TrackedObjectPtr& new_object) {
  // should we estimate measurement if time diff is too small?
  double latest_time = latest_object->object_ptr->timestamp;
  double current_time = new_object->object_ptr->timestamp;
  double time_diff = current_time - latest_time;

  if (fabs(time_diff) < EPSILON_TIME) {
    time_diff = DEFAULT_FPS;
  }
  
  MeasureDetectBboxCenterPointVelocity(new_object, latest_object, time_diff);
  MeasureAnchorPointVelocity(new_object, latest_object, time_diff);
  
  // MeasureBboxCenterVelocity(new_object, latest_object, time_diff);
  // MeasureBboxCornerVelocity(new_object, latest_object, time_diff);
  
  MeasureDetectBboxCornerVelocity(new_object, latest_object, time_diff);

  // MeasurementSelection(latest_object, new_object);
  MeasurementSelection_from_cui(latest_object, new_object, time_diff);
  MeasurementQualityEstimation(latest_object, new_object);
}
void MultiAnchorMotionMeasurement::MeasurementSelection_from_cui(
    const TrackedObjectConstPtr& latest_object, TrackedObjectPtr& new_object, const double& time_diff) {

// =======================================================================
  // [修复 Start] 强制使用“位置差分速度”代替“模型输出速度”
  // =======================================================================
  // 1. 计算基于中心点的差分速度 (Position Diff Velocity)
  // 这是解决无速度模型(Waymo)导致速度坍缩为0的关键
  Eigen::Vector3f calculated_velocity = Eigen::Vector3f::Zero();
  std::vector<Eigen::Vector3f> all_velocity;
  if (time_diff > 1e-4) {
      calculated_velocity = (new_object->center - latest_object->output_center) / time_diff;
  }

  // 2. 将计算出的速度加入候选列表
  all_velocity.emplace_back(calculated_velocity);

  // 3. (可选) 如果你还想保留原来的观测速度（万一以后换了有速度的模型），可以加个判断
  // 如果观测速度模长很小，且计算速度模长较大，说明观测速度不可信(是0)，只用计算速度
  // 这里为了彻底解决问题，直接只用 calculated_velocity，或者把 calculated_velocity 放进去让它参与竞争
  
  // 如果 new_object->measured_detect_bbox_center_velocity 为 0，它会被下面的排序逻辑淘汰
  // 但为了保险，我们直接把计算出的速度作为首选
  new_object->measured_detect_bbox_center_velocity = calculated_velocity; // 覆盖掉那个 0

  // 重新把覆盖后的速度加进去（为了兼容原有逻辑结构）
  // all_velocity.emplace_back(new_object->measured_detect_bbox_center_velocity); // 上面已经emplace了一个，这里不用重复

  // [修复 End]
  // =======================================================================

  // 安全检查：防止候选列表为空导致越界崩溃
  if (all_velocity.empty()) {
      new_object->selected_measured_velocity = latest_object->output_velocity;
      return;
  }

  // --- 以下逻辑保持原样，负责选择与上一帧差异最小的速度 ---

  // 计算与上一帧输出速度的偏差 (Diff)
  for (auto &velocity : all_velocity)
  {
      velocity -= (latest_object->output_velocity); // + latest_object->output_acceleration * time_diff
  }

  // 1. 处理 X 轴：找到 X 方向偏差绝对值最小的候选
  std::sort(all_velocity.begin(), all_velocity.end(), 
      [&](const Eigen::Vector3f &a, const Eigen::Vector3f &b){ 
          return fabs(a.x()) < fabs(b.x());
      });
  
  // 恢复 X 轴绝对速度
  new_object->selected_measured_velocity(0) = all_velocity[0](0) + latest_object->output_velocity.x();


  // 2. 处理 Y 轴：找到 Y 方向偏差绝对值最小的候选
  std::sort(all_velocity.begin(), all_velocity.end(), 
      [&](const Eigen::Vector3f &a, const Eigen::Vector3f &b){ 
          return fabs(a.y()) < fabs(b.y());
      });
  
  // 恢复 Y 轴绝对速度
  new_object->selected_measured_velocity(1) = all_velocity[0](1) + latest_object->output_velocity.y();

  // Z 轴速度通常置 0 或保持不变，视具体需求而定，这里原代码未处理 Z 轴，维持现状
}
void MultiAnchorMotionMeasurement::MeasurementSelection(
    const TrackedObjectConstPtr& latest_object, TrackedObjectPtr& new_object) {
  // Select measured velocity among candidates according motion consistency
  int64_t corner_index = 0;
  float corner_velocity_gain = 0.0f;
  std::vector<float> corner_velocity_gain_norms(4);  // 计算四个角点速度增量的L2范数
  for (int i = 0; i < 4; ++i) {
    corner_velocity_gain_norms[i] =
        static_cast<float>((new_object->measured_corners_velocity[i] -
                            latest_object->output_velocity)
                               .norm());
  }
  // 将四个corner中速度变化最小的增量作为 corner_velocity_gain
  std::vector<float>::iterator corener_min_gain =
      std::min_element(std::begin(corner_velocity_gain_norms),
                       std::end(corner_velocity_gain_norms));
  corner_velocity_gain = *corener_min_gain;
  corner_index = corener_min_gain - corner_velocity_gain_norms.begin();

  // corner_gain, anchor_point_gain, box_center_gain
  std::vector<float> velocity_gain_norms(3);
  velocity_gain_norms[0] = corner_velocity_gain;
  velocity_gain_norms[1] =
      static_cast<float>((new_object->measured_barycenter_velocity -
                          latest_object->output_velocity)
                             .norm());
  velocity_gain_norms[2] = static_cast<float>(
      (new_object->measured_center_velocity - latest_object->output_velocity)
          .norm());

  // 找这四个里面与上一帧速度差值的L2范数最小的作为最终的 selected_measured_velocity
  std::vector<float>::iterator min_gain = std::min_element(
      std::begin(velocity_gain_norms), std::end(velocity_gain_norms));
  int64_t min_gain_pos = min_gain - velocity_gain_norms.begin();
  if (min_gain_pos == 0) {
    new_object->selected_measured_velocity =
        new_object->measured_corners_velocity[corner_index];
  }
  if (min_gain_pos == 1) {
    new_object->selected_measured_velocity =
        new_object->measured_barycenter_velocity;
  }
  if (min_gain_pos == 2) {
    new_object->selected_measured_velocity =
        new_object->measured_center_velocity;
  }
}

void MultiAnchorMotionMeasurement::MeasurementQualityEstimation(
    const TrackedObjectConstPtr& latest_object, TrackedObjectPtr& new_object) {
  // 1. point size diff (only for same sensor)
  // int pre_num = static_cast<int>(latest_object->object_ptr->points->size());
  // int cur_num = static_cast<int>(new_object->object_ptr->points->size());
  // double quality_based_on_point_diff_ratio =
  //             (1.0 - fabs(pre_num - cur_num) / std::max(pre_num, cur_num));
  // 2. association quality
  // double quality_based_on_association_score =
  //     pow(1.0 - new_object->association_score, 2.0);
  // new_object->update_quality = std::min(quality_based_on_association_score,
  //                                       quality_based_on_point_diff_ratio);

  // new_object->update_quality = quality_based_on_point_diff_ratio;
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
