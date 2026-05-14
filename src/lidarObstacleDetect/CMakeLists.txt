cmake_minimum_required(VERSION 3.0.2)
project(lidarObstacleDetect)

# ======================= 1. Core Environment (C++17) =======================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++17 -O3)

add_definitions(-D_FORCE_INLINES) 

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++17")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -std=c++17")
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    list(APPEND CUDA_NVCC_FLAGS "-Xcudafe --diag_suppress=20012")
endif()

# ======================= 2. Find Dependencies =======================
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  visualization_msgs
  tf
  pcl_ros
  pcl_conversions
  autoware_msgs
  msg_gen
  jsk_rviz_plugins
)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Find Pre-compiled SDK Library
find_library(LIDAR_NET_SDK_LIB
    NAMES lidar-net-sdk
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib/lidarnetsdk/build/src
    NO_DEFAULT_PATH
)

if (NOT LIDAR_NET_SDK_LIB)
    message(FATAL_ERROR "Could not find 'liblidar-net-sdk.so'! Please compile SDK first.")
else()
    message(STATUS "Found LidarNet SDK Library: ${LIDAR_NET_SDK_LIB}")
endif()

# ======================= 3. Include Directories =======================
catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs tf pcl_ros msg_gen autoware_msgs jsk_rviz_plugins
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}  # [FIX] Added project root so "lib/fusion/..." works
  include
  src
  lib
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  
  # Internal Modules
  lib/fusion
  lib/tracker
  lib/tracker/lib/interface
  lib/tracker/common
  lib/cluster/euclideanCluster  # [新增]
  lib/ground_detector/patchwork # [新增]
  lib/bounding_box              # [新增]
  # SDK Paths
  lib/lidarnetsdk/include
  lib/lidarnetsdk/src
  lib/lidarnetsdk/src/inference/trt/plugins
  lib/lidarnetsdk/src/common
)

# ======================= 4. Source Files =======================
file(GLOB_RECURSE FUSION_SRCS "lib/fusion/*.cpp" "lib/fusion/*.cu")
file(GLOB_RECURSE TRACKER_SRCS "lib/tracker/*.cpp""lib/tracker/*.cc")
# [新增] 手动添加聚类相关源文件，确保路径准确
set(CLUSTER_SRCS
    lib/cluster/euclideanCluster/euclideanCluster.cpp
    lib/ground_detector/patchwork/patchwork.cpp
    lib/bounding_box/bounding_box.cpp
)
# ======================= 5. Build Node =======================
cuda_add_executable(lidar_detector_node
  src/lidar_detection_track.cpp
  ${FUSION_SRCS}
  ${TRACKER_SRCS}
  ${CLUSTER_SRCS} 
)

# ======================= 6. Link Libraries =======================
target_link_libraries(lidar_detector_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${LIDAR_NET_SDK_LIB}
  nvinfer       
  nvonnxparser  

)

# ======================= 7. Install =======================
install(TARGETS lidar_detector_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)