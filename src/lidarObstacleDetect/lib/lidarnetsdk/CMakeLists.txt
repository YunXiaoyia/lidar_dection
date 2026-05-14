cmake_minimum_required(VERSION 3.22)
project(
  LidarNetSdk 
  VERSION 0.0.1
  DESCRIPTION "SDK for lidar net application, such as lidar detection"
  LANGUAGES CXX CUDA)

option(UNIT_TEST "building unit_test with googletest " on)

if(UNIT_TEST)
  add_definitions(-DUNIT_TEST)
  enable_testing()
endif()
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      "/opt/deepways_master"
      CACHE PATH "My default install prefix" FORCE)
endif()
add_definitions(-DCMAKE_INSTALL_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")

include(3rd/3rdConfig.cmake)
include(cmake/deepways_utils.cmake)
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(CTest)
set(PROJECT_DESCRIPTION "lidar_net_sdk")
if(NOT PACKAGE_HUB)
  deepways_make_package(${PROJECT_NAME} ${PROJECT_VERSION} ${PROJECT_DESCRIPTION})
endif()

# ##############################################################################
# Platform, default NVIDIA
# ##############################################################################
cmake_host_system_information(RESULT HostName QUERY HOSTNAME)
if(${HostName} MATCHES "mdchost")
  set(PLATFORM "mdc")
  add_definitions("-DUSING_ACL")
else()
  set(PLATFORM "nvidia") # default nvidia
  add_definitions("-DUSING_CUDA")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64") # PC
      set(CMAKE_CUDA_ARCHITECTURES
          60
          61
          62
          70
          72
          80
          86)
    else() # arm
      set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72)
    endif()
  endif()
  message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Compile on platform: ${PLATFORM}")

# ##############################################################################
# Global compile options
# ##############################################################################
if(CMAKE_BUILD_TYPE STREQUAL "")
  message("CMAKE_BUILD_TYPE is not set, default as Release")
  set(CMAKE_BUILD_TYPE "Release") # Release or Debug or RelWithDebInfo
else()
  message(STATUS "CMAKE_BUILD_TYPE is set as ${CMAKE_BUILD_TYPE}")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    " ${CMAKE_CXX_FLAGS} -fPIC -pthread -fopenmp -Wall -Wno-error=ignored-attributes -Wno-error=attributes"
)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -pthread -fopenmp")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

link_libraries(stdc++fs)

# ##############################################################################
# Global env
# ##############################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/3rd)

# ##############################################################################
# Modules
# ##############################################################################
if(${PLATFORM} STREQUAL "mdc")
  message(STATUS "Inference with MDC")
elseif(${PLATFORM} STREQUAL "nvidia")
  message(STATUS "Inference with NVIDIA")
  add_definitions("-DINFERENCE_NVIDIA")
  include(3rd/NvidiaConfig.cmake)
endif()

include_directories(include)
include_directories(src)
add_subdirectory(src)

if(UNIT_TEST)
  add_subdirectory(test)
endif()

add_subdirectory(eval)

# ##############################################################################
# Package
# ##############################################################################
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}Config-version.cmake"
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}Config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}")
