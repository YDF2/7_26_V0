add_library(led STATIC
        led_engine.cpp)

target_link_libraries(led pthread)  