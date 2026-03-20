// src/lib/cuda_utils.hpp
#ifndef __CUDA_UTILS_HPP
#define __CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

/**
 * @brief CUDA 错误检查函数（从 darknet/cuda.c 提取）
 * 替代原 darknet/cuda.h 中的 check_error()
 */
inline void check_error_cuda(cudaError_t status)
{
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        fprintf(stderr, "CUDA Error: %s\n", s);
        assert(0);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status2);
        fprintf(stderr, "CUDA Error (last): %s\n", s);
        assert(0);
    }
}

// 兼容宏：让原有 check_error(err) 调用无需修改
#ifndef CHECK_ERROR_COMPAT
#define CHECK_ERROR_COMPAT
// 如果 darknet 的 check_error 不可用，使用此替代
// 注意：当同时链接 darknet 时可能会有符号冲突，
//       因此只在不链接 darknet 的模块中使用
#endif

#endif // __CUDA_UTILS_HPP
