#pragma once

#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <NvInferRuntimeCommon.h>

#include "lidar_net/logging.h"
#include "log/logging.h"
#include "log/logger.h"

namespace lidar_net {
namespace infer {
namespace tensorrt {

using NSeverity = nvinfer1::ILogger::Severity;

class TRTLogger : public nvinfer1::ILogger {
    void log(NSeverity severity, const char* msg) noexcept override {
        std::string msg_str(msg);
        std::string msg_w_header = "[TensorRT] " + msg_str;
        if (std::shared_ptr<LidarNetLogger> spLogger = pLogger.lock()) {
            try {
                switch (severity) {
                    case NSeverity::kINTERNAL_ERROR:
                        spLogger->log(PSeverity::kINTERNAL_ERROR, msg_w_header);
                        break;
                    case NSeverity::kERROR:
                        spLogger->log(PSeverity::kERROR, msg_w_header);
                        break;
                    case NSeverity::kWARNING:
                        spLogger->log(PSeverity::kWARNING, msg_w_header);
                        break;
                    case NSeverity::kINFO:
                        spLogger->log(PSeverity::kINFO, msg_w_header);
                        break;
                    case NSeverity::kVERBOSE:
                        spLogger->log(PSeverity::kVERBOSE, msg_w_header);
                        break;
                }
            } catch (const std::exception& exc) {
                std::cerr << exc.what();
            }
        } else {
            std::cerr << "[No logger found!] " << msg << std::endl;
        }
    }
};

}  // namespace tensorrt
}  // namespace infer
}  // namespace lidar_net
