# seu-unirobot-guidance-v1
东南大学RoboCup Kidsize 校队代码使用指引

This project contains a controller, some debug tools and some scripts for seu-robocup kidszie team. The controller and debug tools are written in c++. The scripts are written in python.

**本教程仅为大纲，安装每个文件时记得查看各自包内的readme文档**

## Build tools
* cmake >= 3.12 [install turtorial](https://www.linuxidc.com/Linux/2018-09/154165.htm)(自带cmake版本较低，需要更新，推荐离线方式安装)
* gcc
* g++
* nvcc

## Dependencies
(in the braces are recommended installation methods)

### c++ libraries
* cuda >= 9.0 (PLZ use [JetPack](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-33/index.html#jetpack/3.3/install.htm%3FTocPath%3D_____3) to install and skip the Jetson TX2 setup)（压缩包内已包含JetPack安装文件）
![][1]
只需要install Host-Ubuntu即可，target下不用管
* cudnn >= 7.0 [install turtorial](https://blog.csdn.net/lucifer_zzq/article/details/76675239)（压缩包内已包含cudnn安装文件）
* opencv >= 3.3.1
* libeigen3-dev 
```
sudo apt-get install libeigen3-dev
```
* libboost-all-dev 
```
sudo apt-get install libboost-all-dev
```
* freeglut3-dev
```
sudo apt-get install build-essential
sudo apt-get install libgl1-mesa-dev
sudo apt-get install libglu1-mesa-dev
sudo apt-get install freeglut3-dev
```
* libv4l-dev
```
sudo apt-get install libv4l-dev
```
* MVSDK (相机驱动) *win下相机的使用见相机驱动.zip*
&nbsp;压缩包内包含MVSDK.7z，解压后按照包内教程继续，如果解压不了
```
sudo apt-get install p7zip-full
```
* qt5-default 
```
sudo apt-get install qt5-default
```
**You can use Jetpack-3.3 to install cuda, opencv and cross compiler tools.**
### python3 libraries
**if there is no pip3 in your computer, you should install python3-pip first**
```
# ubuntu16.04 默认已经安装了python2.7.11和python3.5
sudo apt install python3-pip  # 安装pip3
```
* python3-paramiko
* transitions
* ssh2-python
```
pip3 install paramiko
pip3 install transitions
pip3 install ssh2-python
```

### Tools 
* astyle (for code format)
```
sudo apt-get install astyle
```


## Compile & Run

### Compile for x86_64
```Bash
cd path/to/project
chmod +x x86_64-build.py
./x86_64-build.py
```
* Then you find the executable files in bin/x86_64
* You can run with ./exe_name -h to get infomation about how to use

### Cross compile for aarch64
```Bash
cd path/to/project
./aarch64-build.py
```
* Then you find the executable files in bin/aarch64
* If you want to run program, you should connect with robot, then use the script start_robot.py in bin/

## Recommend OS
* ubuntu 16.04 64bit

## Recommend IDE
* Visual Studio Code(with c++ and python plugin)
* Clion(maybe better with a cmake built-in)


  [1]: https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-33/content/jetpack/images/jetpack_component_mgr.001_600x524.png