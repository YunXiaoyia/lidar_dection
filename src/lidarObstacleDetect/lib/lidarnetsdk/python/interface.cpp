#include <exception>
#include <iostream>
#include <memory>
#include "pillarx/object.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pillarx/pillarx.h>

namespace py = pybind11;
namespace base = pillarx::base;

class StandardStreamLogger : public pillarx::PillarXLogger {
   public:
    void log(pillarx::PillarXLogger::Severity severity, const std::string_view msg) noexcept override {
        try {
            switch (severity) {
                case pillarx::PillarXLogger::Severity::kINTERNAL_ERROR:
                    std::cerr << "[Pillarx][FATAL] " << msg << std::endl;
                    break;
                case pillarx::PillarXLogger::Severity::kERROR:
                    std::cerr << "[Pillarx][ERROR] " << msg << std::endl;
                    break;
                case pillarx::PillarXLogger::Severity::kWARNING:
                    std::cout << "[Pillarx][WARNING] " << msg << std::endl;
                    break;
                case pillarx::PillarXLogger::Severity::kINFO:
                    std::cout << "[Pillarx][INFO] " << msg << std::endl;
                    break;
                case pillarx::PillarXLogger::Severity::kVERBOSE:
                    std::cout << "[Pillarx][DEBUG] " << msg << std::endl;
                    break;
            }
        } catch (const std::exception& exc) {
            std::cerr << exc.what();
        }
    }
};

class PillarxRunner {
    using RowMatrixXd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
    using RowMatrixXf = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
    using RowMatrixXu = Eigen::Matrix<uint8_t, -1, -1, Eigen::RowMajor>;

   public:
    PillarxRunner(const std::string& config, const std::string& config_type)
        : detector_(std::make_unique<pillarx::PillarxDetector>()),
          user_logger_(std::make_shared<StandardStreamLogger>()) {
        bool ret_new = detector_->Init(config, user_logger_, config_type);
        if (!ret_new) {
            user_logger_->log(pillarx::PillarXLogger::Severity::kERROR, "Init Pillarx Detector Failed!");
            throw std::invalid_argument("Init Pillarx Detector Failed!");
        } else {
            std::cout << "Init Pillarx Detector Successfully!\n";
        }
    }

    std::vector<base::BoundingBox> process(const py::EigenDRef<RowMatrixXf> points) {
        auto result = detector_->ProcessingEigenCol(points);
        return result;
    }

   private:
    pillarx::PillarxDetectorPtr detector_;
    std::shared_ptr<StandardStreamLogger> user_logger_;
};

PYBIND11_MODULE(pillarx, m) {
    py::class_<PillarxRunner>(m, "PillarxRunner")
        .def(py::init<const std::string&, const std::string&>())
        .def("process", &PillarxRunner::process);
    py::enum_<base::ObjectType>(m, "ObjectType", py::arithmetic())
        .value("UNKNOWN", base::ObjectType::UNKNOWN)
        /********************************************************/
        .value("VEHICLE", base::ObjectType::VEHICLE)
        .value("CYCLIST", base::ObjectType::CYCLIST)
        .value("PEDESTRIAN", base::ObjectType::PEDESTRIAN)
        .value("BUS", base::ObjectType::BUS)
        /********************************************************/
        .value("ART", base::ObjectType::ART)
        .value("ART_NO_TRAILER", base::ObjectType::ART_NO_TRAILER)
        .value("ART_SEMI_TRAILER", base::ObjectType::ART_SEMI_TRAILER)
        .value("ART_FULL_TRAILER", base::ObjectType::ART_FULL_TRAILER)
        /********************************************************/
        .value("TRUCK", base::ObjectType::TRUCK)
        .value("TRUCK_HEAD", base::ObjectType::TRUCK_HEAD)
        .value("TRAILER", base::ObjectType::TRAILER)
        .value("TRUCK_NO_TRAILER", base::ObjectType::TRUCK_NO_TRAILER)
        .value("TRUCK_SEMI_TRAILER", base::ObjectType::TRUCK_SEMI_TRAILER)
        .value("TRUCK_FULL_TRAILER", base::ObjectType::TRUCK_FULL_TRAILER)
        /********************************************************/
        .value("ALIEN_VEHICLE", base::ObjectType::ALIEN_VEHICLE)
        .value("FORKLIFT", base::ObjectType::FORKLIFT)
        .value("ECCENTRIC_TRUCK_HEAD", base::ObjectType::ECCENTRIC_TRUCK_HEAD)
        .value("MOBILE_CRANE", base::ObjectType::MOBILE_CRANE)
        /********************************************************/
        .value("CONE", base::ObjectType::CONE)
        .value("BARREL", base::ObjectType::BARREL)
        /********************************************************/
        )
        .export_values();
    py::class_<base::BoundingBox>(m, "BoundingBox")
        .def_readwrite("direction", &base::BoundingBox::direction)
        .def_readwrite("theta", &base::BoundingBox::theta)
        .def_readwrite("center", &base::BoundingBox::center)
        .def_readwrite("size", &base::BoundingBox::size)
        .def_readwrite("type_probs", &base::BoundingBox::type_probs)
        .def_readwrite("confidence", &base::BoundingBox::confidence)
        .def_readwrite("corners2d", &base::BoundingBox::corners2d)
        .def_readwrite("type", &base::BoundingBox::type);
}
