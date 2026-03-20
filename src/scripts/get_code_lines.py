#!/usr/bin/env python3
#coding: utf-8

import os
import config


def list_all_files(rootdir):
    _files = []
    lists = os.listdir(rootdir)
    for i in range(0,len(lists)):
           path = os.path.join(rootdir,lists[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files


if __name__ == '__main__': 
    files = list_all_files(config.project_dir+'/src')
    total_count = 0
    for f in files:
        if 'darknet' not in f and os.path.splitext(f)[-1][1:] in ['hpp', 'cpp','c', 'h', 'cu']:
            count = len(open(f, 'r').readlines())
            print('%d\t%s'%(count, f))
            total_count = total_count+count

    print('total lines: %d'%total_count)