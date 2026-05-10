# seu-unirobot

## Description

This project contains a controller, some debug tools and some scripts for seu-robocup kidszie team. The controller and debug tools are written in C++. The scripts are written in Python3.

## Build tools
* cmake >= 3.12
* gcc
* g++
* nvcc

## Dependencies

### C++ libraries
* cuda >= 9.0
* cudnn >= 7.0
* opencv >= 3.3.1
* libeigen3-dev
* libboost-all-dev
* freeglut3-dev
* libv4l-dev
* MVSDK (library for our camera)
* qt5-default

You can use Jetpack-3.3 to install cuda, opencv and cross compiler tools.

### Python3 libraries
* ssh2-python
* if there is no pip3 in your computer, you should install python3-pip first

### Tools
* astyle (for code format)


## Compile & Run

### Compile for x86_64
* cd path/to/project
* ./x86_64-build.py
* Then you find the executable files in bin/x86_64
* You can run with ./exe_name -h to get infomation about how to use

### Cross compile for aarch64
* cd path/to/project
* ./aarch64-build.py
* Then you find the executable files in bin/aarch64
* If you want to run program, you should connect with robot, then use the script start_robot.py in bin/

### Start the robot
* Be sure that your computer is connected to the robot by network
* cd path/to/project/bin, if there is no bin, you should compile first
* Run script: ./start_robot.py id -params
* id is the id of robot, params is start parameters for the app
* if you don't know about the parameters, you can cd into path/to/project/bin/x86_64, run cmd: ./controller -h

## Recommend OS
* ubuntu 16.04 64bit

## Recommend IDE
* Visual Studio Code(with C/C++ and python plugin)
