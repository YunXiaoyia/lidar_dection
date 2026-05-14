

#include "associater/jpda/jpda_associater.h"
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include <set>

#include "Eigen/src/Core/Matrix.h"
#include "common/trunk_time.h"
#include "associater/jpda/filter/jpda.h"


#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

inline float mahalanobis(const Eigen::VectorXf& y,
                    const Eigen::MatrixXf& S)
{
    auto Si = S.inverse();  // Si = inv(S)
    return std::sqrt(y.transpose() * Si * y);  // sqrt(s' * Si * s)
}

inline float logGauss(const Eigen::VectorXf& y,
                      const Eigen::MatrixXf& S) {
    auto Si = S.inverse();  // Si = inv(S)
    float detS = S.determinant();
    float mah = y.transpose() * Si * y;
    std::cout << "detS:" << detS << std::endl;
    std::cout << "Si:" << Si << std::endl;
    // log(exp(-0.5 * (s' * Si * s)) / sqrt(2pi^ns * |S|))  公式3-18  可行联合事件概率
    return(-0.5 * mah - 0.5*((float)y.rows() * (std::log(2*M_PI))+std::log(detS)));
}

inline float gate(int dof) {
    switch (dof) {
    case 1: // 1 dof @ P=0.01
        return 2.576;
        break;
    case 2: // 2 dof @ P=0.01
        return 3.035;
        break;
    case 3: // 3 dof @ P=0.01
        return 3.368;
        break;
    case 4: // 4 dof @ P=0.01
        return 3.644;
        break;
    default:
        cerr << "ERROR - undefined value for gate(" << dof << ")\n";
        std::abort();
    }
}

JPDAAssociater::JPDAAssociater() {

    std::cout << "init JPDA Associater" << std::endl;

    H_.resize(2, 4);
    H_ << 1, 0, 0, 0,
          0, 1, 0, 0;

    A_.resize(4, 4);
    A_ << 1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;

    Q_.resize(4, 4);
    Q_ << 2, 0, 0, 0,
          0, 2, 0, 0,
          0, 0, 2, 0,
          0, 0, 0, 2;

    R_.resize(2, 2);
    R_ << 1, 0, 
          0, 1;
}

bool JPDAAssociater::Init(const toml::node_view<const toml::node>& param_node) {

    // if (!param_node["jpda"].is_table()) {
    //     LOG(WARNING) << "[JPDAAssociater] Dont't find jpda association definition!";
    //     return false;
    // }

    // auto paras = param_node["jpda"];
    // if (!paras["optimizer"].is_value()) {
    //     LOG(WARNING) << "[JPDAAssociater] Dont't find optimizer definition! please check param: optimizer."
    //                  << " use default: hungarian";
    // } else {
    //     auto type = paras["optimizer"].value<std::string>().value();
    //     std::cout << "type:" << type << std::endl;
    //     if (type == "hungarian") {
    //         optimizer_type_ = OptimizerType::HUNGARIAN;
    //     } else if (type == "greedy") {
    //         optimizer_type_ = OptimizerType::GREEDY;
    //     }
    // }

    // if (!paras["match_dist_thresh"].is_value()) {
    //     LOG(WARNING) << "[JPDAAssociater] Dont't find match_dist_thresh definition! please check param: match_dist_thresh."
    //                  << " use default:" << MATCH_DIST_THRESH;
    // } else {
    //     std::cout << "match_dist_thresh:" << MATCH_DIST_THRESH << std::endl;
    //     MATCH_DIST_THRESH = paras["match_dist_thresh"].value<float>().value();
    // }

    return true;
}

// Eigen::Vector3f predicted_anchor_point = last_object->anchor_point;  
//   predicted_anchor_point[0] = predicted_anchor_point[0] + last_object->output_velocity.x() * time_diff;
//   predicted_anchor_point[1] = predicted_anchor_point[1] + last_object->output_velocity.y() * time_diff;

void JPDAAssociater::predict_observation(const TrackletPtr& track, const float& dt, Eigen::VectorXf& p_z, Eigen::MatrixXf& p_variance) {
    p_z.resize(2);
    p_variance.resize(2,2);

    const auto& latest_track = track->GetLatestTrackedObject();
    p_z[0] = latest_track->anchor_point.x() + latest_track->output_velocity.x() * dt;
    p_z[1] = latest_track->anchor_point.y() + latest_track->output_velocity.y() * dt;

    p_variance = latest_track->output_state_covariance.block<2, 2>(0, 0) + Q_.block<2, 2>(0, 0);
}

void JPDAAssociater::Match(const std::vector<common::ObjectPtr> &objects,
                               const std::vector<TrackletPtr> &tracks,
                               AssociationResult& assign_result) {
    
    // Check input objects valid
    assign_result.assignments.clear();
    assign_result.unassigned_measurements.clear();
    assign_result.unassigned_tracks.clear();
    if (objects.empty() || tracks.empty()) {
        assign_result.unassigned_measurements.resize(objects.size());
        assign_result.unassigned_tracks.resize(tracks.size());
        std::iota(assign_result.unassigned_measurements.begin(), assign_result.unassigned_measurements.end(), 0);
        std::iota(assign_result.unassigned_tracks.begin(), assign_result.unassigned_tracks.end(), 0);
        return;
    }

    vector<size_t> znum;
    znum.push_back(objects.size());
    jpda_ = std::make_shared<jpda::JPDA>(znum, tracks.size());

    const size_t M = objects.size(), N = tracks.size();

    Eigen::MatrixXf associate_mat;
    associate_mat.resize(M, N);

    Eigen::VectorXf zp;   // 预测观测值
    Eigen::VectorXf s;    // 预测信息向量
    Eigen::MatrixXf Zp;   // 预测协方差矩阵
    Eigen::MatrixXf S;    // 预测信息协方差矩阵
    for (size_t i = 0; i < M; i++) {

        auto points_eigen = objects[i]->points->getMatrixXfMap(3, 8, 0);
        Eigen::Vector3f measured_anchor_point = points_eigen.rowwise().mean();
        
        for (size_t j = 0; j < N; j++) {
            const auto& latest_object = tracks[j]->GetLatestTrackedObject();
            float dt = objects[i]->timestamp - latest_object->timestamp;
            predict_observation(tracks[j], dt, zp, Zp);

            S = Zp + R_;  // H*P*H' + R    PDA中的公式5，预测信息状态协方差

            s = measured_anchor_point.head(2) - zp; // s:预测信息向量 V(i_k) = z(k) - z(k|k-1)



            if (mahalanobis(s, S) > gate(s.size()) * 2) { // 判断是否在门限内，门限大小根据观测的变量数量进行确定
                continue;
            }

            std::cout << "det id:" << objects[i]->detect_id << "  track id:" << tracks[j]->track_id() << std::endl;
            std::cout << "mahalanobis dis:" << mahalanobis(s, S) << std::endl;

            std::cout << "s:" << s << std::endl;
            std::cout << "S:" << S << std::endl;

            jpda_->Omega[0][i][j+1] = true;  // 构建omega矩阵， true 表示在门限内   注：0只是表示多个传感器，目前只有一个传感器
            jpda_->Lambda[0][i][j+1] = logGauss(s, S);  // 计算可行联合事件的概率

            std::cout << "logGuss:" << jpda_->Lambda[0][i][j+1] << std::endl;
        }
    }

    // compute associations
    jpda_->getAssociations();
    jpda_->getProbabilities();
    vector<jpda::Association> association(1); // 1 表示只有一帧传感器的数据
    jpda_->getMultiNNJPDA(association);

    // data assignment
    jpda::Association::iterator ai, aiEnd = association[0].end();
    std::set<size_t> matatched_tracks, matched_detects;
    for (ai = association[0].begin(); ai != aiEnd; ai++) {

        std::cout<< "ai->t:" << ai->t;
        if (ai->t) {   // not a clutter
            matatched_tracks.insert(ai->t - 1);
            matched_detects.insert(ai->z);
            assign_result.assignments.emplace_back(ai->t - 1, ai->z);
        }
    }
    std::cout << std::endl;

    std::cout << "association result:" << assign_result.assignments.size() << std::endl;

    for (size_t i = 0; i < M; i++) {
        if (matched_detects.count(i)) {
            continue;
        }
        assign_result.unassigned_measurements.emplace_back(i);
    }

    for (size_t i = 0; i < N; i++) {
        if (matatched_tracks.count(i)) {
            continue;
        }
        assign_result.unassigned_tracks.emplace_back(i);
    }
 
    return; 
}



}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
