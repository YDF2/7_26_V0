set(NAME walk_remote)

add_library(${NAME}
        ${NAME}.cpp)

target_link_libraries(${NAME}
        tcp_client
        Qt5::Widgets
        Qt5::Core
        Qt5::Gui)

