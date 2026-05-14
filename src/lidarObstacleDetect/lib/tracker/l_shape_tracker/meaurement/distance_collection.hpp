//
// Created by lgj on 2021/8/12.
//

#pragma once

#include <Eigen/Core>
#include "rotated_iou.hpp"

/**
 * @brief 计算IoU
 *
 * @param vec1
 * @param vec2
 * @param scale
 * @return double
 */
float getIOUDistance(const Eigen::Matrix<float, 2, 4>& bbox1_corners, const Eigen::Matrix<float, 2, 4>& bbox2_corners,
    const Eigen::Vector3f& bbox1_extent, const Eigen::Vector3f& bbox2_extent) {
    const auto area1 = bbox1_extent(0) * bbox1_extent(1);
    const auto area2 = bbox2_extent(0) * bbox2_extent(1);
    if (area1 < 1e-14 || area2 < 1e-14) {
        return 0.0F;
    }

    const auto intersection_area = getIntersectionArea(bbox1_corners, bbox2_corners);
    return (intersection_area / (area1 + area2 - intersection_area));
}

float getIOUDistance(const std::array<Eigen::Vector3f, 4>& bbox1_corners3d, const std::array<Eigen::Vector3f, 4>& bbox2_corners3d,
                     const Eigen::Vector3f& bbox1_extent, const Eigen::Vector3f& bbox2_extent) {
    const auto area1 = bbox1_extent(0) * bbox1_extent(1);
    const auto area2 = bbox2_extent(0) * bbox2_extent(1);
    if (area1 < 1e-14 || area2 < 1e-14) {
        return 0.0F;
    }

    Eigen::Matrix<float, 2, 4> bbox1_corners;
    Eigen::Matrix<float, 2, 4> bbox2_corners;

    bbox1_corners.col(0) = bbox1_corners3d[0].topRows(2);
    bbox1_corners.col(1) = bbox1_corners3d[1].topRows(2);
    bbox1_corners.col(2) = bbox1_corners3d[2].topRows(2);
    bbox1_corners.col(3) = bbox1_corners3d[3].topRows(2);

    bbox2_corners.col(0) = bbox2_corners3d[0].topRows(2);
    bbox2_corners.col(1) = bbox2_corners3d[1].topRows(2);
    bbox2_corners.col(2) = bbox2_corners3d[2].topRows(2);
    bbox2_corners.col(3) = bbox2_corners3d[3].topRows(2);

    const auto intersection_area = getIntersectionArea(bbox1_corners, bbox2_corners);
    return (intersection_area / (area1 + area2 - intersection_area));
}

float getIOUDistance(const std::array<Eigen::Vector3f, 4>& bbox1_corners3d, const Eigen::Matrix<float, 2, 4>& bbox2_corners,
                     const Eigen::Vector3f& bbox1_extent, const Eigen::Vector3f& bbox2_extent) {
    const auto area1 = bbox1_extent(0) * bbox1_extent(1);
    const auto area2 = bbox2_extent(0) * bbox2_extent(1);
    if (area1 < 1e-14 || area2 < 1e-14) {
        return 0.0F;
    }

    Eigen::Matrix<float, 2, 4> bbox1_corners;
    bbox1_corners.col(0) = bbox1_corners3d[0].topRows(2);
    bbox1_corners.col(1) = bbox1_corners3d[1].topRows(2);
    bbox1_corners.col(2) = bbox1_corners3d[2].topRows(2);
    bbox1_corners.col(3) = bbox1_corners3d[3].topRows(2);

    const auto intersection_area = getIntersectionArea(bbox1_corners, bbox2_corners);
    return (intersection_area / (area1 + area2 - intersection_area));
}

/**
 * @brief compute point num distance for given track & object
 *
 * @param last_obj point number
 * @param new_obj point number
 * @return distance
 */
float pointNumDistance(const int& last_obj, const int& new_obj) {
    float point_num_dist =
        static_cast<float>(fabs(last_obj - new_obj)) * 0.1f / static_cast<float>(std::max(last_obj, new_obj));
    return point_num_dist;
}

/**
 * @brief 据算bbox 大小的差距
 *
 * @param last_bbox_extent
 * @param new_bbox_extent
 * @return distance
 */
float bboxSizeDistance(const Eigen::Vector3f& last_bbox_extent, const Eigen::Vector3f& new_bbox_extent) {
    float last_size = last_bbox_extent[0] * last_bbox_extent[1];
    float new_size = new_bbox_extent[0] * new_bbox_extent[1];
    float bbox_size_dist = fabs(last_size - new_size) / std::max(last_size, new_size);
    return bbox_size_dist;
}

