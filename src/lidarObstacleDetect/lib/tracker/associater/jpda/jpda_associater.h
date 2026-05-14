//
// Created by jie.gong on 22-07-21.
//
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include "Eigen/src/Core/Matrix.h"
#include "common/base.h"

#include "associater/associater.h"

#include "associater/jpda/filter/jpda.h"


#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


class JPDAAssociater : public BaseAssociater {
  
private:
    /// ////////////////////
    /// Hyper Parameters
    /// ////////////////////

    // Limit maxmum associate distance
    float MATCH_DIST_THRESH = 4.0;

    // Limit associate boundary
    float MATCH_DIST_BOUND = 100.0;

    // Limit maxmum center point associate distance
    float ASSOCIATE_CENTER_DIST_THRESH = 30.0;

public:
    JPDAAssociater();
    ~JPDAAssociater() = default;

    bool Init(const toml::node_view<const toml::node>& param_node) override;

    void Match(const std::vector<common::ObjectPtr> &objects,
               const std::vector<TrackletPtr> &tracks,
               AssociationResult& assign_result) override;

    std::string Name() const override { return "JPDAAssociater"; };

protected:
    // @brief: compute association matrix
    // @params [in]: maintained tracks for matching
    // @params [in]: new detected objects for matching
    // @params [out]: matrix of association distance
    // void ComputeAssociateMatrix(const std::vector<TrackletPtr> &tracks,
    //                             const std::vector<common::ObjectPtr> &new_objects,
    //                             SecureMat<float> *association_mat);

    // void Associate(const std::vector<TrackletPtr> &tracks,
    //                const std::vector<size_t> &track_idxs,
    //                const std::vector<common::ObjectPtr> &new_objects,
    //                const std::vector<size_t> &object_idxs,
    //                AssociationResult& assign_results);
    void predict_observation(const TrackletPtr& track, const float& dt, Eigen::VectorXf& p_z, Eigen::MatrixXf& p_variance);

protected:
    
    std::shared_ptr<jpda::JPDA> jpda_;

    Eigen::MatrixXf H_;
    Eigen::MatrixXf A_;

    Eigen::MatrixXf Q_;
    Eigen::MatrixXf R_;

    

};  // class JPDAAssociater



}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

