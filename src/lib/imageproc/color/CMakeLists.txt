add_library(color STATIC
            color.cpp)

target_link_libraries(color
            opencv_core
            opencv_imgproc)   