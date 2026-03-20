include_directories(./)
include_directories(./player)
add_subdirectory(options)
add_subdirectory(player)
add_subdirectory(drivers)


set(TARGET_NAME controller)

add_executable(${TARGET_NAME}
                main.cpp)

target_link_libraries(${TARGET_NAME}
                robot
                parser
                controller_options
                player)

install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/${CMAKE_SYSTEM_PROCESSOR}")
