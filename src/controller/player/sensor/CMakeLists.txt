add_library(sensor STATIC
            button.cpp
            camera.cpp
            motor.cpp
            imu.cpp
            gamectrl.cpp
            hear.cpp
            hipnuc_dec.c
            serial_port.c
            cmd_read.c)
            
target_link_libraries(sensor
            MVSDK
            server
            dynamixel
            gpio
            boost_system)

# ZED-mini backend (optional)
if (EXISTS "/usr/local/zed/include/sl/Camera.hpp" AND EXISTS "/usr/local/zed/lib/libsl_zed.so")
    target_compile_definitions(sensor PRIVATE USE_ZED_BACKEND)
    target_include_directories(sensor PRIVATE /usr/local/zed/include)
    target_link_libraries(sensor /usr/local/zed/lib/libsl_zed.so)

    # ZED->Vision stage 1 needs CPU resize + color convert in camera.cpp
    target_link_libraries(sensor opencv_core opencv_imgproc)
endif()