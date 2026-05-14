//
// Created by jie.gong on 22-07-01.
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include "common/base.h"
#include "../../lidarnetsdk/src/common/toml.hpp"

#include "common/tracked_object.h"
#include "simple_track/tracklet.h"



#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

typedef std::pair<size_t, size_t> TrackMeasurmentPair;

struct AssociationResult {
    std::vector<TrackMeasurmentPair> assignments;
    std::vector<size_t> unassigned_tracks;
    std::vector<size_t> unassigned_measurements;
}; 


class BaseAssociater {

public:
    BaseAssociater() = default;
    ~BaseAssociater() = default;

    virtual bool Init(const toml::node_view<const toml::node>& param_node) = 0;

    virtual void Match(const std::vector<common::ObjectPtr> &objects,
                       const std::vector<TrackletPtr> &tracks,
                       AssociationResult& assign_result) = 0;

    virtual std::string Name() const = 0;
};  // class BaseAssociater

using AssociaterPtr = std::shared_ptr<BaseAssociater>;

class AssociaterFactory {

public:
    static AssociaterPtr MakeAssociater(const toml::node_view<const toml::node>& param_node);
};



}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

