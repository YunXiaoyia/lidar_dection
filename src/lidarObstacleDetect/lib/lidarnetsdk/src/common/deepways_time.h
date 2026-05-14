#pragma once

#include <limits>
#include <string>

#include <chrono>
#include "lidar_net/logging.h"
#include "log/logger.h"
#include "log/logging.h"

namespace lidar_net {


/// @brief 获得当前UTC时间/秒
inline double getTime_s() {
    auto time_now = std::chrono::system_clock::now();
    auto duration_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch());
    return duration_in_ms.count() / 1000000.0;
}

using std::chrono::high_resolution_clock;

class TicToc {
public:
    TicToc() = default;

    // no-thread safe.
    void Tic() { start_time_ = TicToc::Now(); }

    // return the elapsed time,
    // also output msg and time in glog.
    // automatically start a new timer.
    // no-thread safe.
    double Toc(const std::string& msg) {
        end_time_ = TicToc::Now();
        double elapsed_time = (end_time_ - start_time_) / 1e6;
        PLOG_INFO << "[" << msg << "] " << elapsed_time << " ms";

        // start new timer.
        start_time_ = end_time_;
        return elapsed_time;
    }

    static uint64_t Now() {
        auto now = high_resolution_clock::now();
        auto nano_time_point =
            std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
        auto epoch = nano_time_point.time_since_epoch();
        uint64_t now_nano =
            std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();

        return now_nano;
    }

private:
    uint64_t start_time_;
    uint64_t end_time_;

    TicToc(const TicToc &) = delete;
    TicToc &operator=(const TicToc &) = delete;
};

};

#define PERF_BLOCK_START()     \
  lidar_net::TicToc _timer_; \
  _timer_.Tic()
#define PERF_BLOCK_END(msg) _timer_.Toc(msg)


