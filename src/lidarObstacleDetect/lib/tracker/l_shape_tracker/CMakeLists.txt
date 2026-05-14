add_compile_options(-march=native)
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    add_compile_options(-mfma)
endif ()

aux_source_directory(./ LSHAPE_DIR_SRCS)
aux_source_directory(./association LSHAPE_DIR_SRCS)
aux_source_directory(./measurement LSHAPE_DIR_SRCS)


add_library(l_shape_tracker SHARED
        ${LSHAPE_DIR_SRCS}
        ) 

target_link_libraries(l_shape_tracker
        PkgConfig::JIAN
        )

# INSTALL(TARGETS tracker LIBRARY DESTINATION lib)

##################################################################################################
## install
##################################################################################################
install(TARGETS l_shape_tracker
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

