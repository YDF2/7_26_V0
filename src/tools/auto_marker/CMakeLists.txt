# auto_marker 需要包含 controller/player 以找到 vision/trt_detector.hpp
include_directories(${CMAKE_SOURCE_DIR}/src/controller/player)

find_package(Boost REQUIRED COMPONENTS filesystem system)

add_executable(auto_marker
    main.cpp
    ${CMAKE_SOURCE_DIR}/src/controller/player/vision/trt_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/controller/player/vision/trt_preprocess.cu)

target_link_libraries(auto_marker 
    Boost::filesystem
    Boost::system
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    nvinfer
    nvinfer_plugin
    cudart)

install(TARGETS auto_marker RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/${CMAKE_SYSTEM_PROCESSOR}")