add_library(field_monitor
        field_monitor.cpp)

target_link_libraries(field_monitor
        tcp_client
        Qt5::Widgets
        opencv_core
        pthread)
