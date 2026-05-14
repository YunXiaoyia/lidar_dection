add_compile_options(-march=native)
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    add_compile_options(-mfma)
endif ()


add_subdirectory(l_shape_tracker)

aux_source_directory(./ DIR_SRCS)
aux_source_directory(./common DIR_SRCS)
aux_source_directory(./associater DIR_SRCS)
aux_source_directory(./associater/distance_measurement DIR_SRCS)
aux_source_directory(./associater/greedy DIR_SRCS)
aux_source_directory(./associater/hungarian DIR_SRCS)
aux_source_directory(./associater/jpda DIR_SRCS)
aux_source_directory(./associater/jpda/filter DIR_SRCS)
aux_source_directory(./motion_filter/multi_anchor DIR_SRCS)
aux_source_directory(./motion_filter/l_shape DIR_SRCS)
aux_source_directory(./shape_filter/moving_average DIR_SRCS)
aux_source_directory(./simple_track DIR_SRCS)

add_library(highway_lidar_tracker SHARED
        ${DIR_SRCS}
        src/lib/tracker/motion_filter/multi_anchor/velocity_utils.cpp
        )

target_link_libraries(highway_lidar_tracker
        PkgConfig::JIAN
        l_shape_tracker
        )

# INSTALL(TARGETS tracker LIBRARY DESTINATION lib)

##################################################################################################
## install
##################################################################################################
install(TARGETS highway_lidar_tracker
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION include/lib/highway_lidar_tracker
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
  )