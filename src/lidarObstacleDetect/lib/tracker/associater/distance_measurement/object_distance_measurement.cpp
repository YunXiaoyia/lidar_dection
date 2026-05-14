
#include "associater/distance_measurement/object_distance_measurement.h"
#include "associater/distance_measurement/general_distance_cal.h"

#include <common/lidar_perception_log.h>

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

// location dist weight, direction dist weight, bbox size dist weight,
// point num dist weight, histogram dist weight, centroid shift dist weight
// bbox iou dist weight, box tail dist_weight
ObjectDistanceMeasurement::ObjectDistanceMeasurement(){
  this->kDefaultWeight_ = {
      0.3f,   // location
      0.1f,   // direction
      0.1f,   // bbox size
      0.1f,   // point num
      0.2f,   // histogram
      0.01f,  // centroid
      0.1f,   // iou
      0.2f    // box tail
      };

    this->bbox_iou_match_threshold_ = 0.7f;
}

bool ObjectDistanceMeasurement::Init() {

  return true;
}

float ObjectDistanceMeasurement::ComputeDistance(
    const common::ObjectPtr& object,
    const TrackletPtr& track) {

  float distance = 0.f;
  float delta = 1e-10f;

  double current_time = object->timestamp;
  const auto& latest_object = track->GetLatestTrackedObject();

  // double time_diff = current_time - track->last_tracked_time();
  double time_diff = current_time - latest_object->timestamp;

  // TLOG_INFO << "track id:" << latest_object->track_id << "  detect id:" << object->id << "  time_diff: " << time_diff;
  // auto&& log = COMPACT_GOOGLE_LOG_INFO;

  if (kDefaultWeight_[0] > delta) {
    float dis0 = kDefaultWeight_[0] * LocationDistance(latest_object, object, time_diff);
    distance += dis0;
    // TLOG_INFO  << "LocationDistance dis0: " << dis0;
  }
  if (kDefaultWeight_[1] > delta) {
      float dis1 = kDefaultWeight_[1] * DirectionDistance(latest_object, object, time_diff);
      distance += dis1;
      // TLOG_INFO << "DirectionDistance dis1: " << dis1;
  }
  if (kDefaultWeight_[2] > delta) {
    float dis2 = kDefaultWeight_[2] * BboxSizeDistance(latest_object, object, time_diff);
    distance += dis2;
    // TLOG_INFO << "BboxSizeDistance dis2: " << dis2;
  }
  if (kDefaultWeight_[3] > delta) {
    float dis3 = kDefaultWeight_[3] * PointNumDistance(latest_object, object, time_diff);
    distance += dis3;
    // TLOG_INFO << "PointNumDistance dis3: " << dis3;
  }
  // if (weights->at(4) > delta) {
  //   distance +=
  //       weights->at(4) * HistogramDistance(latest_object, track->predict_.state,
  //                                          object);
  // }
  if (kDefaultWeight_[5] > delta) {
    float dis5 = kDefaultWeight_[5] * CentroidShiftDistance(latest_object, object, time_diff);
    distance += dis5;
    // TLOG_INFO << "CentroidShiftDistance dis5: " << dis5;
  }
  if (kDefaultWeight_[6] > delta) {
    float dis6 = kDefaultWeight_[6] * BboxIouDistance(latest_object, object,
                                      time_diff, bbox_iou_match_threshold_);
    distance += dis6;
    // TLOG_INFO << "BboxIouDistance dis6: " << dis6;
  }

  if(kDefaultWeight_[7] > delta){
    float dist7 = kDefaultWeight_[7] * BboxTailDistance(latest_object, object, time_diff);
    distance += dist7;
    // TLOG_INFO << "BboxTailDistance dist7: " << dist7;
  }

  // TLOG_INFO << "dist_all: " << distance;
  return distance;
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
