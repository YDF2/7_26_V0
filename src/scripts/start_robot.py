#!/usr/bin/env python3
#coding: utf-8

import common
import sys
import config
import datetime
import threading
import time
import signal
import ssh_connection

if __name__ == '__main__': # 若正在直接运行本文件（即python start_robot.py)
    if not common.check_argv(sys.argv):
        common.print_error('no enough arguments')
        common.print_info('%s id params')
        exit(2)

    robot_id = sys.argv[1]
    if not common.check_id(robot_id):
        common.print_error('please check the robot id')
        exit(3)
    j=2
    for i in range(2,len(sys.argv)):
        if '-j' in sys.argv[i]: # j 代表多线程
            j=int(sys.argv[i][2:])
            break

    if not common.build_project(True, j):
        common.print_error('build error, please check code')
        exit(4)

    args = common.parse_argv(sys.argv)
    ip_address = common.get_ip(robot_id)

    if not common.check_net(ip_address):
        common.print_error('can not connect to host, please check network')
        exit(6)
        
    ssh_client = ssh_connection.ssh_connection(ip_address, config.ssh_port, config.username, config.password)

    if not common.compress_files(ssh_client):
        common.print_error('compress files error, please check')
        exit(5)

    ssh_client.upload(config.project_dir+'/bin/'+config.compress_file_name, config.remote_dir+config.compress_file_name)
    
    def signal_handler(sig,frame):
        if sig == signal.SIGINT:
            ssh_client.exec_command('kill -2 $(pidof %s)'%(config.exec_file_name), False)

    signal.signal(signal.SIGINT,signal_handler)
    
    cmd = 'cd %s; tar zxmf %s -C %s; cd %s; ./%s %s'\
          %(config.remote_dir, config.compress_file_name, config.target_dir, config.target_dir, config.exec_file_name, args)
    ssh_client.exec_command(cmd)
