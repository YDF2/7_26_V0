#!/usr/bin/env python3

import os
import shutil
import sys

if __name__ == '__main__': 
    build_dir = 'aarch64-build'
    t = '2'
    if len(sys.argv) == 2:
        t = sys.argv[1]
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)

    # Clear stale configure cache to avoid keeping CMAKE_CUDA_COMPILER-NOTFOUND.
    cache_file = os.path.join(build_dir, 'CMakeCache.txt')
    cmake_files_dir = os.path.join(build_dir, 'CMakeFiles')
    if os.path.exists(cache_file):
        os.remove(cache_file)
    if os.path.isdir(cmake_files_dir):
        shutil.rmtree(cmake_files_dir)

    # On Orin NX (native aarch64), do not force cross-compilation mode.
    is_native_aarch64 = (os.uname().machine == 'aarch64')
    cross_flag = 'OFF' if is_native_aarch64 else 'ON'
    cmd = (
        'cd %s; '
        'cmake -D CROSS=%s -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..; '
        'make install -j%s'
    ) % (build_dir, cross_flag, t)
    os.system(cmd)
