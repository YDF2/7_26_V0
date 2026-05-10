add_executable(auto_marker
    main.cpp
    ../../controller/player/vision/trt_detector.cpp)

target_include_directories(auto_marker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/controller/player/vision)

target_link_libraries(auto_marker 
    boost_system 
    boost_filesystem 
    imageproc
    nvinfer
    nvinfer_plugin
    cudart
    opencv_core
    opencv_imgcodecs
    opencv_imgproc)

install(TARGETS auto_marker RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/${CMAKE_SYSTEM_PROCESSOR}")