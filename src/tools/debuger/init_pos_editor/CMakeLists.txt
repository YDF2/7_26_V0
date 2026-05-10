set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(NAME init_pos_editor)

add_library(${NAME}
    init_pos_editor.cpp
    init_pos_form.cpp
    init_pos_service.cpp
)

target_link_libraries(${NAME} Qt5::Widgets)
