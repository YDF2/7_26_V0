add_library(imageproc STATIC
        imageproc.cpp
        imageproc.cu)

target_link_libraries(imageproc
        cudart
        opencv_core
        opencv_imgcodecs
        opencv_imgproc)
