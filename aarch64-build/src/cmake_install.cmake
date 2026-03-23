# Install script for directory: /home/seurobot2/Downloads/7_26_V0/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/seurobot2/Downloads/7_26_V0/bin")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/seurobot2/Downloads/7_26_V0/bin/aarch64/data")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/seurobot2/Downloads/7_26_V0/bin/aarch64" TYPE DIRECTORY FILES "/home/seurobot2/Downloads/7_26_V0/src/data")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/seurobot2/Downloads/7_26_V0/bin/code_format.py;/home/seurobot2/Downloads/7_26_V0/bin/common.py;/home/seurobot2/Downloads/7_26_V0/bin/config.py;/home/seurobot2/Downloads/7_26_V0/bin/download.py;/home/seurobot2/Downloads/7_26_V0/bin/exec.py;/home/seurobot2/Downloads/7_26_V0/bin/get_code_lines.py;/home/seurobot2/Downloads/7_26_V0/bin/install_auto_run.py;/home/seurobot2/Downloads/7_26_V0/bin/set_network.py;/home/seurobot2/Downloads/7_26_V0/bin/ssh_connection.py;/home/seurobot2/Downloads/7_26_V0/bin/start_robot.py;/home/seurobot2/Downloads/7_26_V0/bin/uninstall_auto_run.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/seurobot2/Downloads/7_26_V0/bin" TYPE PROGRAM PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ GROUP_WRITE FILES
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/code_format.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/common.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/config.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/download.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/exec.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/get_code_lines.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/install_auto_run.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/set_network.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/ssh_connection.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/start_robot.py"
    "/home/seurobot2/Downloads/7_26_V0/src/scripts/uninstall_auto_run.py"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/seurobot2/Downloads/7_26_V0/aarch64-build/src/lib/cmake_install.cmake")
  include("/home/seurobot2/Downloads/7_26_V0/aarch64-build/src/controller/cmake_install.cmake")

endif()

