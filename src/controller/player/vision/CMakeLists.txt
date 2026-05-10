add_library(vision
        vision.cpp
        trt_detector.cpp)

target_link_libraries(vision
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        imageproc
        nvinfer
        nvinfer_plugin
        server
        SL
        pthread
        rt
        stdc++
        cudart)
