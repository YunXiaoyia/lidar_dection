#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "logging.h"
#include "object.h"

namespace lidar_net {

enum class StatusCode {
    Succeed = 0,
    InputError = 1
};

struct ProcessStatus {
    std::string log;
    StatusCode status_code;

    explicit ProcessStatus(const StatusCode& status_code_in) {
        status_code = status_code_in;
    }

    ProcessStatus(const StatusCode& status_code_in,
                      const std::string& log_in) {
        status_code = status_code_in;
        log = log_in;
    }
};

class LidarNetDetectorImpl;

class LidarNetDetector {
    using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstRefRow = const Eigen::Ref<const RowMatrixXf>&;
    using ConstRef = const Eigen::Ref<const Eigen::MatrixXf>&;

   public:
    LidarNetDetector();
    ~LidarNetDetector();

    // Init the detector with the config file
    bool Init(const std::string_view config_path, std::shared_ptr<LidarNetLogger> logger,
              const std::string& config_type);

    // Raw C data input
    std::vector<base::BoundingBox> Processing(void* points, const size_t points_count, ProcessStatus* status = nullptr);
    void Processing(void* points, const size_t points_count, std::vector<base::BoundingBox>& bboxes, ProcessStatus* status = nullptr);

    // Eigen data input (RowMajor)
    std::vector<base::BoundingBox> ProcessingEigenRow(ConstRefRow points, ProcessStatus* status = nullptr);
    void ProcessingEigenRow(ConstRefRow points, std::vector<base::BoundingBox>& bboxes, ProcessStatus* status = nullptr);

    // Eigen data input (ColMajor)
    std::vector<base::BoundingBox> ProcessingEigenCol(ConstRef points, ProcessStatus* status = nullptr);
    void ProcessingEigenCol(ConstRef points, std::vector<base::BoundingBox>& bboxes, ProcessStatus* status = nullptr);

   private:
    std::unique_ptr<LidarNetDetectorImpl> impl_;
};

using LidarNetDetectorPtr = std::shared_ptr<LidarNetDetector>;
using LidarNetDetectorUniPtr = std::unique_ptr<LidarNetDetector>;

}  // namespace lidar_net
