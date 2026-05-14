cmake_minimum_required(VERSION 3.0.2)
project(multi_lidar_fusion)

## Compile as C++14
add_compile_options(-std=c++14)

# ++++++++++++++++ 添加CUDA支持 ++++++++++++++++
find_package(CUDA REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  sensor_msgs
  message_filters
)

find_package(PCL REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# --- (核心修改) ---
# 将 .cpp 和 .cu 文件一起交给 cuda_add_library
# 这样 nvcc 编译器会同时处理 C++ 和 CUDA 代码
cuda_add_library(${PROJECT_NAME} STATIC
  multi_lidar_fusion.cpp
  cuda_transform.cu
)

# 链接依赖库
# (注意：我们现在链接 ${PROJECT_NAME}，而不是 cuda_transform)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
)
