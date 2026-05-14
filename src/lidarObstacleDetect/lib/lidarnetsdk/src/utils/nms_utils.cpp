
#include <Eigen/Core>

#include "nms_utils.h"

namespace lidar_net {

float cross_2d(const Eigen::Vector2f& A, const Eigen::Vector2f& B) { return A(0) * B(1) - A(1) * B(0); }

int getIntersectionPoints(const Corners& corners1, const Corners& corners2, Intersects& intersection_points) {
    Corners vec1, vec2;
    for (int i = 0; i < 4; ++i) {
        vec1.col(i) = corners1.col((i + 1) % 4) - corners1.col(i);
        vec2.col(i) = corners2.col((i + 1) % 4) - corners2.col(i);
    }

    int num = 0;
    constexpr float EPS = 1e-5;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const auto det = cross_2d(vec2.col(j), vec1.col(i));

            if (std::abs(det) <= 1e-14) {
                continue;
            }

            const Eigen::Vector2f vec12 = corners2.col(j) - corners1.col(i);
            const auto t1 = cross_2d(vec2.col(j), vec12) / det;
            const auto t2 = cross_2d(vec1.col(i), vec12) / det;

            if (-EPS < t1 && t1 < 1.0f + EPS && -EPS < t2 && t2 < 1.0f + EPS) {
                intersection_points.col(num++) = corners1.col(i) + vec1.col(i) * t1;
            }
        }
    }
    // Check for vertices of rect1 inside rect2
    {
        const auto& AB = vec2.col(0);
        const auto& DA = vec2.col(3);
        auto ABdotAB = AB.dot(AB);
        auto ADdotAD = DA.dot(DA);
        for (int i = 0; i < 4; i++) {
            // assume ABCD is the rectangle, and P is the point to be judged
            // P is inside ABCD iff. P's projection on AB lies within AB
            // and P's projection on AD lies within AD

            const auto AP = corners1.col(i) - corners2.col(0);
            const auto APdotAB = AP.dot(AB);
            const auto APdotAD = -AP.dot(DA);

            if ((APdotAB > -EPS) && (APdotAD > -EPS) && (APdotAB < ABdotAB + EPS) && (APdotAD < ADdotAD + EPS)) {
                intersection_points.col(num++) = corners1.col(i);
            }
        }
    }
    // Reverse the check - check for vertices of rect2 inside rect1
    {
        const auto& AB = vec1.col(0);
        const auto& DA = vec1.col(3);
        auto ABdotAB = AB.dot(AB);
        auto ADdotAD = DA.dot(DA);
        for (int i = 0; i < 4; i++) {
            const auto AP = corners2.col(i) - corners1.col(0);
            const auto APdotAB = AP.dot(AB);
            const auto APdotAD = -AP.dot(DA);

            if ((APdotAB > -EPS) && (APdotAD > -EPS) && (APdotAB < ABdotAB + EPS) && (APdotAD < ADdotAD + EPS)) {
                intersection_points.col(num++) = corners2.col(i);
            }
        }
    }
    return num;
}

int convexHullGraham(const Intersects& p, const int num_in, Intersects& q, bool shift_to_zero) {
    assert(num_in >= 2);

    // Step 1:
    // Find point with minimum y
    // if more than 1 points have the same minimum y, pick the one with the minimum x.
    int t = 0;
    for (int i = 1; i < num_in; ++i) {
        if (p(1, i) < p(1, t) || (p(1, i) == p(1, t) && p(0, i) < p(0, t))) {
            t = i;
        }
    }

    // Step 2:
    // Subtract starting point from every points (for sorting in the next step)
    q = p.colwise() - p.col(t);

    // Swap the starting point to position 0
    q.col(0).swap(q.col(t));

    // Step 3:
    // Sort point 1 ~ num_in according to their relative cross-product values (essentially sorting according to angles)
    // If the angles are the same, sort according to their distance to origin
    std::qsort(q.col(1).data(), num_in - 1, 2 * sizeof(*q.data()), [](const void* a, const void* b) {
        Eigen::Map<const Eigen::Vector2f> A(static_cast<const float*>(a));
        Eigen::Map<const Eigen::Vector2f> B(static_cast<const float*>(b));
        float temp = cross_2d(A, B);
        if (std::abs(temp) < 1e-6) {
            if (A.dot(A) < B.dot(B))
                return -1;
            else
                return 1;
        } else {
            if (temp > 0.0F)
                return -1;
            else
                return 1;
        }
    });

    // compute distance to origin after sort, since the points are now different.
    // Eigen::Matrix<float, 24, 1> dist = (q.array().square()).colwise().sum();
    Eigen::VectorXf dist = (q.array().square()).colwise().sum();

    // Step 4:
    // Make sure there are at least 2 points (that don't overlap with each other) in the stack
    int k;  // index of the non-overlapped second point
    for (k = 1; k < num_in; ++k) {
        if (dist(k) > 1e-8) {
            break;
        }
    }
    if (k == num_in) {
        // We reach the end, which means the convex hull is just one point
        q.col(0) = p.col(t);
        return 1;
    }
    q.col(1) = q.col(k);
    int m = 2;  // 2 points in the stack

    // Step 5:
    // Finally we can start the scanning process.
    // When a non-convex relationship between the 3 points is found
    // (either concave shape or duplicated points),
    // we pop the previous point from the stack
    // until the 3-point relationship is convex again, or
    // until the stack only contains two points
    for (int i = k + 1; i < num_in; ++i) {
        while (1 < m) {
            const Eigen::Vector2f q1 = q.col(i) - q.col(m - 2);
            const Eigen::Vector2f q2 = q.col(m - 1) - q.col(m - 2);
            if (q1(0) * q2(1) < q2(0) * q1(1))
                break;
            else
                --m;
        }

        q.col(m++) = q.col(i);
    }

    // Step 6 (Optional):
    // In general sense we need the original coordinates, so we
    // need to shift the points back (reverting Step 2)
    // But if we're only interested in getting the area/perimeter of the shape
    // We can simply return.
    // In this current situation of tkdl, we do not need revert to the original coords.
    if (!shift_to_zero) {
        q.colwise() += p.col(t);
    }

    return m;
}

float polygonArea(const Intersects& q, const int m) {
    if (m <= 2) {
        return 0;
    }

    float area = 0.0f;
    for (int i = 1; i < m - 1; ++i) {
        area += std::abs(cross_2d(q.col(i) - q.col(0), q.col(i + 1) - q.col(0)));
    }

    return area * 0.5F;
}

float getIntersectionArea(const Corners& corners1, const Corners& corners2) {
    Intersects intersection_points, ordered_points;
    const int num = getIntersectionPoints(corners1, corners2, intersection_points);

    if (num <= 2) {
        return 0.0F;
    }

    const int num_convex = convexHullGraham(intersection_points, num, ordered_points, true);

    return polygonArea(ordered_points, num_convex);
}

// float getRotatedIOU(const BBox2D& bbox1_raw, const BBox2D& bbox2_raw) {
float getRotatedIOU(const lidar_net::base::BoundingBox& bbox1_raw, const lidar_net::base::BoundingBox& bbox2_raw) {
    const auto area1 = bbox1_raw.size(0) * bbox1_raw.size(1);
    const auto area2 = bbox2_raw.size(0) * bbox2_raw.size(1);
    if (area1 < 1e-14 || area2 < 1e-14) {
        return 0.0F;
    }

    const auto intersection_area = getIntersectionArea(bbox1_raw.corners2d, bbox2_raw.corners2d);
    return (intersection_area / (area1 + area2 - intersection_area));
}

float getOverlapRate(const lidar_net::base::BoundingBox& bbox1_raw, const lidar_net::base::BoundingBox& bbox2_raw) {
    const auto area1 = bbox1_raw.size(0) * bbox1_raw.size(1);
    const auto area2 = bbox2_raw.size(0) * bbox2_raw.size(1);
    if (area1 < 1e-14 || area2 < 1e-14) {
        return 0.0F;
    }

    const auto intersection_area = getIntersectionArea(bbox1_raw.corners2d, bbox2_raw.corners2d);
    return intersection_area / (std::min(area1, area2) + 1e-6);
}

void nms(std::vector<lidar_net::base::BoundingBox>& vec_bbox, const float iou_thresh) {
    const size_t sz_bbox = vec_bbox.size();
    if (sz_bbox <= 1) {
        return;
    }
    std::vector<std::pair<float, size_t>> score_index_vec(sz_bbox);
    for (size_t i = 0; i < sz_bbox; ++i) {
        score_index_vec[i].first = vec_bbox[i].confidence_iou;
        score_index_vec[i].second = i;
    }

    const auto con = [](const std::pair<float, size_t>& p1, const std::pair<float, size_t>& p2) {
        return p1.first < p2.first;
    };
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(), con);
    std::vector<size_t> keep_indices;
    while (!score_index_vec.empty()) {
        const size_t idx = score_index_vec.back().second;
        bool is_keep = true;
        for (size_t i = 0; i < keep_indices.size(); ++i) {
            const size_t keep_idx = keep_indices[i];
            const float overlap = getRotatedIOU(vec_bbox[idx], vec_bbox[keep_idx]);
            is_keep = std::abs(overlap) < iou_thresh;
            if (!is_keep) {
                break;
            }
        }
        if (is_keep) {
            keep_indices.push_back(idx);
        }
        score_index_vec.pop_back();
    }
    std::vector<lidar_net::base::BoundingBox> bbox2d_keeped;
    for (size_t i = 0; i < keep_indices.size(); ++i) {
        size_t idx_keep = keep_indices.at(i);
        bbox2d_keeped.emplace_back(vec_bbox.at(idx_keep));
    }
    vec_bbox.swap(bbox2d_keeped);
}

}  // namespace lidar_net
