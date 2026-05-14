# plugins
add_subdirectory(plugins)

# inference
add_library(trt_inference STATIC trt_infer.cpp trt_logger.cpp util.cpp)
target_link_libraries(trt_inference logger ${TRT_LIBRARIES}
                      CUDA::cublas CUDA::cudart)

target_compile_options(trt_inference PRIVATE "-march=native")
target_compile_options(trt_inference PRIVATE "-Wno-deprecated-declarations")
