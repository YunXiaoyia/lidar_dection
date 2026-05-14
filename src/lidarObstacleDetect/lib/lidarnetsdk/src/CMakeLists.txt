add_library(logger STATIC log/logger.cpp)

if(${PLATFORM} STREQUAL "mdc")
  add_subdirectory(inference/mdc)
elseif(${PLATFORM} STREQUAL "nvidia")
  add_subdirectory(inference/trt)
endif()

set(LIDAR_NET_SOURCES)
set(LIDAR_NET_LISTS lidar_net utils)
foreach(LIDAR_NET_ITER ${LIDAR_NET_LISTS})
  include_directories(${LIDAR_NET_ITER})
  add_subdirectory(${LIDAR_NET_ITER})
endforeach(LIDAR_NET_ITER)

set(DETECTOR_EXPORT_MAP ${CMAKE_CURRENT_SOURCE_DIR}/exports.map)

# ##############################################################################
# SHARED LIBRARY
# ##############################################################################
add_library(lidar-net-sdk SHARED ${LIDAR_NET_SOURCES})

target_link_libraries(lidar-net-sdk logger ryml)

if(${PLATFORM} STREQUAL "nvidia")
  target_link_libraries(lidar-net-sdk trt_inference)
endif()

target_compile_options(lidar-net-sdk PRIVATE "-march=native")
target_compile_options(lidar-net-sdk PRIVATE "-Wno-deprecated-declarations")
set_target_properties(
  lidar-net-sdk
  PROPERTIES
    LINK_FLAGS
    "-Wl,--exclude-libs,ALL -Wl,--version-script=${DETECTOR_EXPORT_MAP} -Wl,--no-undefined"
)

# ##############################################################################
# Install
# ##############################################################################
install(
  TARGETS lidar-net-sdk
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/lidar_net
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h")
