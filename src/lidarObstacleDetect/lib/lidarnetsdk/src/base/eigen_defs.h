
//
// Created by jie.gong on 22-02-17.
//

#pragma once

#include <functional>
#include <map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace lidar_net {
namespace base {

// Using STL Containers with Eigen:
// https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
template <class EigenType>
using EigenVector = std::vector<EigenType, Eigen::aligned_allocator<EigenType>>;

template <typename T, class EigenType>
using EigenMap = std::map<T, EigenType, std::less<T>, Eigen::aligned_allocator<std::pair<const T, EigenType>>>;

using EigenVector3dVec = EigenVector<Eigen::Vector3d>;
using EigenAffine3dVec = EigenVector<Eigen::Affine3d>;

}  // namespace base
}  // namespace lidar_net
