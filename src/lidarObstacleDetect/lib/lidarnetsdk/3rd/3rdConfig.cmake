# Eigen
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
include_directories(${EIGEN3_INCLUDE_DIR})

# rapid yaml
add_library(ryml 3rd/ryml/ryml_all.cpp)
include_directories(3rd/ryml)