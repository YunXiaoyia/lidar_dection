
if(UNIT_TEST)

    # Find OpenCV
    find_package(OpenCV REQUIRED)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "Not found OpenCV")
    endif ()

    # spdlog
    include_directories(BEFORE SYSTEM 3rd)

    # Catch2
    add_library(Catch2 3rd/catch_amalgamated.cpp)

    # lidar_net test
    file(GLOB LIDAR_NET_TEST_SRCS test*.cpp)
    add_executable(test_lidar_net ${LIDAR_NET_TEST_SRCS})
    target_link_libraries(test_lidar_net lidar-net-sdk ${OpenCV_LIBS})

    target_compile_options(test_lidar_net PRIVATE "-march=native")
    target_compile_options(test_lidar_net PRIVATE "-Wno-deprecated-declarations")

    # install
    install(
        TARGETS test_lidar_net
        EXPORT ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
        DESTINATION bin/
        FILES_MATCHING
        PATTERN "*.npy")
    install(
        DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/config
        DESTINATION bin/
        FILES_MATCHING
        PATTERN "*.toml")

endif(UNIT_TEST)