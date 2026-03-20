#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief Letterbox 预处理 CUDA kernel
 *
 * 功能：将 BGR uint8 HWC 图像一步转换为 RGB float32 CHW 的 letterbox 输入
 *   1. 双线性插值缩放（保持宽高比）
 *   2. BGR → RGB 颜色通道转换
 *   3. [0,255] → [0,1] 归一化
 *   4. HWC → CHW 排列
 *   5. 空白区域填充 114/255 ≈ 0.447
 *
 * @param src     输入图像 GPU 指针 (BGR uint8, HWC, src_w × src_h)
 * @param dst     输出张量 GPU 指针 (RGB float32, CHW, dst_w × dst_h)
 * @param src_w   原图宽度
 * @param src_h   原图高度
 * @param dst_w   目标宽度（网络输入，如 640）
 * @param dst_h   目标高度（网络输入，如 640）
 * @param scale   缩放比例 = min(dst_w/src_w, dst_h/src_h)
 * @param pad_x   水平填充偏移
 * @param pad_y   垂直填充偏移
 */
__global__ void letterbox_preprocess_kernel(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale, int pad_x, int pad_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 目标图像 x 坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 目标图像 y 坐标

    if (x >= dst_w || y >= dst_h) return;

    int area = dst_w * dst_h;
    int dst_idx = y * dst_w + x;  // CHW 中的像素索引

    // 有效区域范围
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);

    float r, g, b;

    // 判断当前像素是否在有效缩放区域内
    if (x >= pad_x && x < pad_x + new_w &&
        y >= pad_y && y < pad_y + new_h)
    {
        // 映射回原图坐标（双线性插值）
        float src_xf = (float)(x - pad_x) / scale;
        float src_yf = (float)(y - pad_y) / scale;

        // 双线性插值的四个邻居
        int x0 = (int)src_xf;
        int y0 = (int)src_yf;
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        // 边界 clamp
        if (x1 >= src_w) x1 = src_w - 1;
        if (y1 >= src_h) y1 = src_h - 1;
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;

        float dx = src_xf - x0;
        float dy = src_yf - y0;
        float w00 = (1.0f - dx) * (1.0f - dy);
        float w01 = dx * (1.0f - dy);
        float w10 = (1.0f - dx) * dy;
        float w11 = dx * dy;

        // 读取 BGR 四邻域（HWC 排列）
        int idx00 = (y0 * src_w + x0) * 3;
        int idx01 = (y0 * src_w + x1) * 3;
        int idx10 = (y1 * src_w + x0) * 3;
        int idx11 = (y1 * src_w + x1) * 3;

        // BGR → RGB 并归一化
        b = (w00 * src[idx00 + 0] + w01 * src[idx01 + 0] +
             w10 * src[idx10 + 0] + w11 * src[idx11 + 0]) / 255.0f;
        g = (w00 * src[idx00 + 1] + w01 * src[idx01 + 1] +
             w10 * src[idx10 + 1] + w11 * src[idx11 + 1]) / 255.0f;
        r = (w00 * src[idx00 + 2] + w01 * src[idx01 + 2] +
             w10 * src[idx10 + 2] + w11 * src[idx11 + 2]) / 255.0f;
    }
    else
    {
        // 填充区域：灰色 114/255
        r = g = b = 114.0f / 255.0f;
    }

    // 写入 CHW 格式：R 平面 → G 平面 → B 平面
    dst[0 * area + dst_idx] = r;
    dst[1 * area + dst_idx] = g;
    dst[2 * area + dst_idx] = b;
}

/**
 * @brief Letterbox 预处理入口函数（供 C++ 调用）
 */
extern "C"
void cudaLetterboxPreprocess(
    const unsigned char* dev_bgr_src,
    float* dev_chw_dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale, int pad_x, int pad_y,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    letterbox_preprocess_kernel<<<grid, block, 0, stream>>>(
        dev_bgr_src, dev_chw_dst,
        src_w, src_h, dst_w, dst_h,
        scale, pad_x, pad_y);
}
