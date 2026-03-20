add_library(scan STATIC
        scan_engine.cpp)

target_link_libraries(scan pthread)  