# Find the CUDNN libraries
#
# The following variables are optionally searched for defaults CUDNN_ROOT: Base
# directory where CUDNN is found CUDNN_INCLUDE_DIR: Directory where CUDNN header
# is searched for CUDNN_LIBRARY_PATH: Directory where CUDNN library is searched
# for CUDNN_STATIC: Are we looking for a static library? (default: no)
#
# The following are set after configuration is done: CUDNN_FOUND
# CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES
#

include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT
    $ENV{CUDNN_ROOT_DIR}
    CACHE PATH "Folder containing NVIDIA cuDNN")
if(DEFINED $ENV{CUDNN_ROOT_DIR})
  message(
    WARNING "CUDNN_ROOT_DIR is deprecated. Please set CUDNN_ROOT instead.")
endif()
list(APPEND CUDNN_ROOT $ENV{CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_ROOT})

# Compatible layer for CMake <3.12. CUDNN_ROOT will be accounted in for
# searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${CUDNN_ROOT})

set(CUDNN_INCLUDE_DIR
    $ENV{CUDNN_INCLUDE_DIR}
    CACHE PATH "Folder containing NVIDIA cuDNN header files")

find_path(
  CUDNN_INCLUDE_DIRS cudnn.h
  HINTS ${CUDNN_INCLUDE_DIR}
  PATH_SUFFIXES cuda/include cuda include)

set(CUDNN_LIBNAME "cudnn")
set(CUDNN_LIBNAME_STATIC "libcudnn_static.a")

set(CUDNN_LIBRARY_PATH
    $ENV{CUDNN_LIBRARY_PATH}
    CACHE PATH "Path to the cudnn library file (e.g., libcudnn.so)")
# if(CUDNN_LIBRARY_PATH MATCHES ".*cudnn_static.a" AND NOT CUDNN_STATIC)
# message( WARNING "CUDNN_LIBRARY_PATH points to a static library
# (${CUDNN_LIBRARY_PATH}) but CUDNN_STATIC is OFF." ) endif()

find_library(
  CUDNN_LIBRARIES ${CUDNN_LIBNAME}
  PATHS ${CUDNN_LIBRARY_PATH}
  PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

# find_library(
#   CUDNN_LIBRARIES_STATIC ${CUDNN_LIBNAME_STATIC}
#   PATHS ${CUDNN_LIBRARY_PATH}
#   PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

# find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARIES
#                                   CUDNN_LIBRARIES_STATIC CUDNN_INCLUDE_DIRS)

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARIES CUDNN_INCLUDE_DIRS)

mark_as_advanced(CUDNN_ROOT CUDNN_INCLUDE_DIR CUDNN_LIBRARY_PATH)
