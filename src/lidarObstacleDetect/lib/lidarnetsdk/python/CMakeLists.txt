# ##############################################################################
# Project
# ##############################################################################
cmake_minimum_required(VERSION 3.22)
project(
  PillarXPy
  VERSION 0.2.3
  DESCRIPTION "PillarX Python"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    " ${CMAKE_CXX_FLAGS} -fPIC -pthread -fopenmp -Wall -Wno-error=ignored-attributes -Wno-error=attributes"
)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -pthread -fopenmp")
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

# ##############################################################################
# Find Packages
# ##############################################################################

# Pthread
find_package(Threads REQUIRED)

# Python3
set(Python3_ROOT_DIR
    /usr/bin/python3
    CACHE STRING "")
message(STATUS "Python3_ROOT_DIR ${Python3_ROOT_DIR}")
find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
include_directories(${EIGEN3_INCLUDE_DIR})

# pybind11
message(STATUS "PYBIND11_INCLUDE_PATH:" ${PYBIND11_INCLUDE_PATH})
message(STATUS "PYBIND11_PACKAGE_PATH:" ${PYBIND11_PACKAGE_PATH})
find_package(PYBIND11 NAMES pybind11 REQUIRED PATHS ${PYBIND11_PACKAGE_PATH})
include_directories(${PYBIND11_INCLUDE_PATH})

# PillarXSDK
find_package(
  PillarXSDK
  0.2.3
  EXACT
  REQUIRED
  VERSION
  PATHS
  /opt/pillarx_sdk/share/cmake/PillarXSDK)
message(STATUS "PillarXSDK version: ${PillarXSDK_VERSION}")
message(STATUS "PillarXSDK libraries: ${PillarXSDK_LIBRARIES}")
message(STATUS "PillarXSDK include dirs: ${PillarXSDK_INCLUDE_DIRS}")
include_directories(${PillarXSDK_INCLUDE_DIRS})

# ##############################################################################
# Python Interface
# ##############################################################################

pybind11_add_module(pillarx interface.cpp)
target_link_libraries(pillarx PRIVATE ${PillarXSDK_LIBRARIES})
