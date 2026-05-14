/*
 * @Author: tianxiaosen Tream733@163.com
 * @Date: 2025-07-02 10:06:25
 * @LastEditors: tianxiaosen Tream733@163.com
 * @LastEditTime: 2025-07-02 10:37:42
 * @FilePath: /lidarnetsdk/include/lidar_net/logging.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once

#include <string_view>

namespace lidar_net {

/**
 * @brief A parody of Nvinfer1::ILogger
 */
class LidarNetLogger {
   public:
    /**
     * @brief The severity corresponding to a log message.
     */
    enum class Severity : int {
        kINTERNAL_ERROR = 0,  /// Internal error has occurred. Execution is unrecoverable.
        kERROR = 1,           /// Application error has occurred.
        kWARNING = 2,  /// Application error has been discovered. TensorRT has recovered or fallen back to a default.
        kINFO = 3,     /// Informational messages with instructional information.
        kVERBOSE = 4,  /// Verbose messages with debugging information.
    };

    /**
     * @brief A callback implemented by the application to handle logging messages;
     *
     * @param severity The severity of the message.
     * @param msg The log message, null terminated.
     */
    virtual void log(Severity severity, const std::string_view msg) noexcept = 0;

    virtual ~LidarNetLogger() {}
};

}  // namespace lidar_net
