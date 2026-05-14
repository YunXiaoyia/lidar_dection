# add dsvt
include_directories(dsvt)

# add flatformer
include_directories(flatformer)

file(GLOB_RECURSE SRCS *.cpp)
message("srcs in lidar_net folder: ${SRCS}")

set(LIDAR_NET_SOURCES ${LIDAR_NET_SOURCES} ${SRCS})
set(LIDAR_NET_SOURCES ${LIDAR_NET_SOURCES} PARENT_SCOPE)
