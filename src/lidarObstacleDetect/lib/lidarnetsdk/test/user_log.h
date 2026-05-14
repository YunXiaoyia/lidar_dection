#include "spdlog/spdlog.h"
#include "lidar_net/lidar_net_det.h"

// init log
class SpdlogUserLogger : public lidar_net::LidarNetLogger {
    void log(lidar_net::LidarNetLogger::Severity severity, const std::string_view msg) noexcept override {
        try {
            switch (severity) {
                case lidar_net::LidarNetLogger::Severity::kINTERNAL_ERROR:
                    spdlog::critical("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kERROR:
                    spdlog::error("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kWARNING:
                    spdlog::warn("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kINFO:
                    spdlog::info("[lidar_net] {}", msg);
                    break;
                case lidar_net::LidarNetLogger::Severity::kVERBOSE:
                    spdlog::debug("[lidar_net] {}", msg);
                    break;
            }
        } catch (const std::exception& exc) {
            std::cerr << exc.what();
        }
    }
};