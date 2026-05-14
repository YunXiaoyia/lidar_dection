include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT
    $ENV{CUDNN_ROOT_DIR}
    CACHE PATH "Folder containing NVIDIA cuDNN")
if(DEFINED $ENV{CUDNN_ROOT_DIR})
  message(
    WARNING "CUDNN_ROOT_DIR is deprecated. Please set CUDNN_ROOT instead.")
endif()
list(APPEND CUDNN_ROOT $ENV{CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_ROOT})
# TRT
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  find_library(LIB_NVONNXPARSER nvonnxparser)
  find_library(LIB_NVPARSERS nvparsers)
  find_library(LIB_NVINFER nvinfer)
  find_library(LIB_NVINFER_PLUGIN nvinfer_plugin)
  # find_library(LIB_NVONNXPARSER_STATIC nvonnxparser_static)
  # find_library(LIB_NVPARSERS_STATIC nvparsers_static)
  # find_library(LIB_NVINFER_STATIC nvinfer_static)
  set(TRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu")
else()
  set(TensorRT_DIR $ENV{TensorRT_DIR})
  set(TRT_LIB_DIR ${TensorRT_DIR}/lib)
  set(TRT_INCLUDE_DIRS ${TensorRT_DIR}/include)
  find_library(
    LIB_NVONNXPARSER REQUIRED
    NAMES nvonnxparser
    PATHS ${TRT_LIB_DIR})
  find_library(
    LIB_NVPARSERS REQUIRED
    NAMES nvparsers
    PATHS ${TRT_LIB_DIR})
  find_library(
    LIB_NVINFER REQUIRED
    NAMES nvinfer
    PATHS ${TRT_LIB_DIR})

  find_library(
     LIB_NVINFER_PLUGIN REQUIRED
     NAMES nvinfer_plugin
     PATHS ${TRT_LIB_DIR})


  # find_library(
  #   LIB_NVONNXPARSER_STATIC REQUIRED
  #   NAMES nvonnxparser_static
  #   PATHS ${TRT_LIB_DIR})
  # find_library(
  #   LIB_NVPARSERS_STATIC REQUIRED
  #   NAMES nvparsers_static
  #   PATHS ${TRT_LIB_DIR})
  # find_library(
  #   LIB_NVINFER_STATIC REQUIRED
  #   NAMES nvinfer_static
  #   PATHS ${TRT_LIB_DIR})
endif()

set(TRT_LIBRARIES ${LIB_NVONNXPARSER} ${LIB_NVPARSERS} ${LIB_NVINFER} ${LIB_NVINFER_PLUGIN})
# set(TRT_LIBRARIES_STATIC
#     ${LIB_NVONNXPARSER_STATIC}
#     ${LIB_NVPARSERS_STATIC}
#     ${LIB_NVINFER_STATIC})
set(TRT_LIBRARIES ${LIB_NVONNXPARSER} ${LIB_NVPARSERS} ${LIB_NVINFER}  ${LIB_NVINFER_PLUGIN})

# find_package_handle_standard_args(TensorRT DEFAULT_MSG TRT_LIBRARIES
#                                   TRT_LIBRARIES_STATIC TRT_INCLUDE_DIRS)

find_package_handle_standard_args(TensorRT DEFAULT_MSG TRT_LIBRARIES TRT_INCLUDE_DIRS)