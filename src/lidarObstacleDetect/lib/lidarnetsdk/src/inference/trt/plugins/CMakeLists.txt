set(PLUGIN_EXPORT_MAP ${CMAKE_CURRENT_SOURCE_DIR}/exports.map)

set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

set(PLUGIN_LISTS
    pillarScatterPlugin
    scatterPlugin
    voxelGeneratorPlugin
    gridSamplerPlugin
    multiViewVoxelGeneratorPlugin
    scatterMaxPlugin
    pillarScatterCamPlugin)

include_directories(common)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
  include_directories(${PLUGIN_ITER})
  add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

add_subdirectory(common)

list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

# trt_plugins_base
add_library(trt_plugins_base STATIC ${PLUGIN_SOURCES})

target_link_libraries(trt_plugins_base ${TRT_LIBRARIES} ${CUDNN_LIBRARIES}
                      CUDA::cudart CUDA::cublas)

target_compile_options(trt_plugins_base PRIVATE -Wno-deprecated-declarations -w)

# ##############################################################################
# SHARED LIBRARY
# ##############################################################################

# initLibNvInferPlugins
add_library(trt_plugins STATIC api/InferPlugin.cpp)

target_link_libraries(
  trt_plugins
  trt_plugins_base
  ${TRT_LIBRARIES}
  ${CUDNN_LIBRARIES}
  CUDA::cublas
  CUDA::cudart
  ${CMAKE_DL_LIBS})

target_compile_options(trt_plugins PRIVATE -Wno-deprecated-declarations -w)
set_target_properties(
  trt_plugins
  PROPERTIES
    LINK_FLAGS
    "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,--no-undefined"
)

# REGISTER_TENSORRT_PLUGIN
add_library(trtexec_plugins SHARED api/RegisterPlugin.cpp)

target_link_libraries(
  trtexec_plugins
  trt_plugins_base
  ${TRT_LIBRARIES}
  ${CUDNN_LIBRARIES}
  CUDA::cublas
  CUDA::cudart
  ${CMAKE_DL_LIBS})

target_compile_options(trtexec_plugins PRIVATE -Wno-deprecated-declarations -w)
set_target_properties(
  trtexec_plugins
  PROPERTIES
    LINK_FLAGS
    "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,--no-undefined"
)

# ##############################################################################
# ## install
# ##############################################################################
install(
  TARGETS trtexec_plugins
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
