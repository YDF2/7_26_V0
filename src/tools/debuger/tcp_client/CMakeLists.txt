add_library(tcp_client STATIC
            tcp_client.cpp)
            
set(BOOST_FIND_MINIMUM_COMPONENTS serialization system filesystem thread program_options date_time timer chrono regex)
find_package(Boost REQUIRED COMPONENTS ${BOOST_FIND_MINIMUM_COMPONENTS})
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	link_directories(${Boost_LIBRARY_DIRS})
endif()



target_link_libraries(tcp_client
            Boost::system
            pthread)