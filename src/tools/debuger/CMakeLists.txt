set(TARGET_NAME debuger)

include_directories(${Qt5Widgets_INCLUDE_DIRS})
# set(BOOST_ROOT /usr/local/boost158) #设置boost库搜索路径
# set(BOOST_FIND_MINIMUM_COMPONENTS serialization system filesystem thread program_options date_time timer chrono regex)
# find_package(Boost 1.58.0 EXACT REQUIRED COMPONENTS ${BOOST_FIND_MINIMUM_COMPONENTS})
# if(Boost_FOUND)
# 		include_directories(${Boost_INCLUDE_DIRS})
# 		link_directories(${Boost_LIBRARY_DIRS})
# endif()


include_directories(.)

add_subdirectory(ui)
add_subdirectory(tcp_client)
add_subdirectory(image_monitor)
add_subdirectory(image_debuger)
add_subdirectory(camera_setter)
add_subdirectory(action_debuger)
add_subdirectory(action_monitor)
add_subdirectory(walk_remote)
add_subdirectory(joint_revise)
add_subdirectory(team_monitor)


add_executable(${TARGET_NAME}
                main.cpp
                ${TARGET_NAME}.cpp)

target_link_libraries(${TARGET_NAME}
                parser
                image_monitor
                image_debuger
                camera_setter
                action_debuger
                action_monitor
                walk_remote
                joint_revise
                team_monitor
                tcp_client
                glut
                GLU
                GL
                Qt5::Widgets 
                Qt5::Core
                Qt5::Gui)

install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/${CMAKE_SYSTEM_PROCESSOR}")
