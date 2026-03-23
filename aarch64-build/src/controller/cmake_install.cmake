# Install script for directory: /home/seurobot2/Downloads/7_26_V0/src/controller

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
  if(EXISTS "$ENV{DESTDIR}/home/seurobot2/Downloads/7_26_V0/bin/aarch64/controller" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/seurobot2/Downloads/7_26_V0/bin/aarch64/controller")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/seurobot2/Downloads/7_26_V0/bin/aarch64/controller"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/seurobot2/Downloads/7_26_V0/bin/aarch64/controller")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/seurobot2/Downloads/7_26_V0/bin/aarch64" TYPE EXECUTABLE FILES "/home/seurobot2/Downloads/7_26_V0/aarch64-build/src/controller/controller")
  if(EXISTS "$ENV{DESTDIR}/home/seurobot2/Downloads/7_26_V0/bin/aarch64/controller" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/seurobot2/Downloads/7_26_V0/bin/aarch64/controller")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/seurobot2/Downloads/7_26_V0/bin/aarch64/controller"
         OLD_RPATH "/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:/usr/local/zed/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/seurobot2/Downloads/7_26_V0/bin/aarch64/controller")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/seurobot2/Downloads/7_26_V0/aarch64-build/src/controller/options/cmake_install.cmake")
  include("/home/seurobot2/Downloads/7_26_V0/aarch64-build/src/controller/player/cmake_install.cmake")
  include("/home/seurobot2/Downloads/7_26_V0/aarch64-build/src/controller/drivers/cmake_install.cmake")

endif()

