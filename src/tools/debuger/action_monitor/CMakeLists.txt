set(NAME action_monitor)

add_library(${NAME}
        ${NAME}.cpp)

target_link_libraries(${NAME}
        ui
        robot
        tcp_client
        Qt5::Widgets
        Qt5::Core
        Qt5::Gui)

