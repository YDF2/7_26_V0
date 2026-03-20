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