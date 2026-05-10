#!/usr/bin/env python3

import os
import sys
import time

if __name__ == '__main__':
    comment = time.localtime(time.time())
    repo = "local"

    if len(sys.argv) >= 2:
        comment = sys.argv[1]
    if len(sys.argv) >= 3:
        repo = sys.argv[2]
    
    cmd = 'git add src README.md aarch64-build.py x86_64-build.py CMakeLists.txt .gitignore git.py'
    os.system(cmd)

    cmd = 'git commit -m "{}"'.format(comment)
    os.system(cmd)

    if repo == "remote":
        cmd = 'git push origin master'
        os.system(cmd)
    
