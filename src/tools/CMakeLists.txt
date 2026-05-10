find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5Core CONFIG REQUIRED)
find_package(Qt5Gui CONFIG REQUIRED)
set(CMAKE_AUTOMOC ON)

add_subdirectory(auto_marker)
add_subdirectory(debuger)
