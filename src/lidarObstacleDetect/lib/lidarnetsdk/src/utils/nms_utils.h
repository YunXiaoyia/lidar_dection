#pragma once

#include <Eigen/Core>

#include "lidar_net/object.h"

namespace lidar_net {

using Corners = Eigen::Matrix<float, 2, 4>;
using Intersects = Eigen::Matrix<float, 2, 24>;

float cross_2d(const Eigen::Vector2f& A, const Eigen::Vector2f& B);
int getIntersectionPoints(const Corners& corners1, const Corners& corners2, Intersects& intersection_points);

int convexHullGraham(const Intersects& p, const int num_in, Intersects& q, bool shift_to_zero = false);
float polygonArea(const Intersects& q, const int m);
float getIntersectionArea(const Corners& corners1, const Corners& corners2);

float getRotatedIOU(const lidar_net::base::BoundingBox& bbox1_raw, const lidar_net::base::BoundingBox& bbox2_raw);

void nms(std::vector<lidar_net::base::BoundingBox>& vec_bbox, const float iou_thresh);

float getOverlapRate(const lidar_net::base::BoundingBox& bbox1_raw, const lidar_net::base::BoundingBox& bbox2_raw);

}  // namespace lidar_net
