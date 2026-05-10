set(NAME image_monitor)

add_library(${NAME}
        ${NAME}.cpp)

target_link_libraries(${NAME}
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        ui
        tcp_client
        Qt5::Widgets
        Qt5::Core
        Qt5::Gui)

