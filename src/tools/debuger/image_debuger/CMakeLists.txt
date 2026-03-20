set(NAME image_debuger)

add_library(${NAME}
        ${NAME}.cpp)

target_link_libraries(${NAME}
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        ui
        imageproc
        Qt5::Widgets
        Qt5::Core
        Qt5::Gui)

