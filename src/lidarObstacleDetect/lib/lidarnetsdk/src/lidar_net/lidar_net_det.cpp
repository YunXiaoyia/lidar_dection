#include <memory>

#include "lidar_net/lidar_net_det.h"
#include "lidar_net_det_impl.h"
#include "utils/config_convert.h"
#include "dsvt/dsvt_detector_impl.h"
#include "flatformer/flatformer_detector_impl.h"

namespace lidar_net {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstRefRow = const Eigen::Ref<const RowMatrixXf>&;
using ConstRef = const Eigen::Ref<const Eigen::MatrixXf>&;
using VecBBox = std::vector<base::BoundingBox>;

LidarNetDetector::LidarNetDetector() {}

LidarNetDetector::~LidarNetDetector() = default;

bool LidarNetDetector::Init(const std::string_view config_path, std::shared_ptr<LidarNetLogger> logger,
                           const std::string& config_type) {
    pLogger = logger;

    auto param_node_json_str = convert2json(config_path, config_type);

    std::stringstream os;
    const nlohmann::json config = nlohmann::json::parse(param_node_json_str);
    
    const auto& algorithm = config["algorithm"]; // TODO: 异常检查

    // create impl:
    if (algorithm == "dsvt") {
        impl_.reset(new DSVTDetectorImpl());
    } else if (algorithm == "flatformer") {
        impl_.reset(new FlatFormerDetectorImpl());
    } else {
        PLOG_ERROR << "[Detector] algorithm [" << algorithm << "] is not supported!";
        return false;
    }
    PLOG_INFO << "[Detector] algorithm [" << algorithm << "] is created!";

    // init impl:
    return impl_->Init(param_node_json_str, logger, "json");
}

VecBBox LidarNetDetector::ProcessingEigenCol(ConstRef points, ProcessStatus* status) {
    VecBBox bboxes;
    ProcessingEigenCol(points, bboxes, status);
    return bboxes;
}

void LidarNetDetector::ProcessingEigenCol(ConstRef points, std::vector<base::BoundingBox>& bboxes, ProcessStatus* status) {

    impl_->Processing(points, bboxes, status);
}

VecBBox LidarNetDetector::ProcessingEigenRow(ConstRefRow points, ProcessStatus* status) {
    VecBBox bboxes;
    ProcessingEigenRow(points, bboxes, status);
    return bboxes;
}

void LidarNetDetector::ProcessingEigenRow(ConstRefRow points, std::vector<base::BoundingBox>& bboxes, ProcessStatus* status) {
    impl_->Processing(points, bboxes, status, false);
}

}  // namespace lidar_net