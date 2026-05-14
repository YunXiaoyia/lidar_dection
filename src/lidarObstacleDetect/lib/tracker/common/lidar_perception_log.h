#pragma once

#include <iostream>
#include <string>

// 使用标准输出替代 ROS 宏，彻底规避头文件冲突问题
// ROS 节点运行时会自动捕获 std::cout 到 /rosout

// 定义日志宏 (支持流式操作: TLOG_INFO << "message";)
#define TLOG_INFO  std::cout << "[INFO] [Tracker] "
#define TLOG_WARN  std::cerr << "[WARN] [Tracker] "
#define TLOG_ERROR std::cerr << "[ERROR] [Tracker] "
#define TLOG_DEBUG std::cout << "[DEBUG] [Tracker] "
