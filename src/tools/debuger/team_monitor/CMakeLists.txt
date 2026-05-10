set(NAME team_monitor)

add_library(${NAME}
        state_monitor.cpp
        ${NAME}.cpp)

target_link_libraries(${NAME}
        ui
        boost_system
        pthread
        Qt5::Widgets
        Qt5::Core
        Qt5::Gui)
