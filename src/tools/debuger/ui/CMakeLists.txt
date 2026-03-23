include_directories(${Qt5Widgets_INCLUDE_DIRS})

add_library(ui STATIC
            image_label.cpp
            robot_gl.cpp)

target_link_libraries(ui
            robot
            parser
            opencv_core
            opencv_imgproc
            tcp_client
            glut
            GLU
            GL
            Qt5::Widgets
            Qt5::Core
            Qt5::Gui)