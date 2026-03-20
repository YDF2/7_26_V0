#ifndef __TRT_DETECTOR_HPP
#define __TRT_DETECTOR_HPP

#include <string>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include "model.hpp"  // object_det 结构体

/**
 * @brief TensorRT YOLOv8 检测器封装类
 *
 * 职责：
 *   1. 加载 TensorRT .engine 序列化引擎
 *   2. 执行 letterbox 预处理（GPU CUDA kernel）
 *   3. TensorRT GPU 推理
 *   4. YOLOv8 后处理（坐标解码 + 阈值过滤 + NMS）
 *   5. 输出与原 Darknet 接口完全兼容的 object_det
 *
 * 设计原则：
 *   - 与 Vision 类解耦，可独立测试
 *   - 所有 GPU 缓冲由本类管理，Vision 无需关心
 *   - detect() 接口对外行为与原 Darknet 流程等效
 */
class TRTDetector
{
public:
    TRTDetector();
    ~TRTDetector();

    /**
     * @brief 加载 TensorRT 引擎文件
     * @param engine_path .engine 文件路径
     * @param num_classes 类别数（本项目固定为 2）
     * @return 是否加载成功
     */
    bool load(const std::string& engine_path, int num_classes);

    /**
     * @brief 执行检测（预处理 + 推理 + 后处理）
     *
     * 替代原先的:
     *   cudaResizePacked() + cudaBGR2RGBfp() + network_predict() +
     *   get_network_boxes() + do_nms_sort() + 解析循环 + free_detections()
     *
     * @param dev_bgr     GPU 上的 BGR uint8 图像（dev_undis_）
     * @param img_w       原图宽度（w_，通常 640）
     * @param img_h       原图高度（h_，通常 480）
     * @param ball_dets   [输出] 足球检测结果
     * @param post_dets   [输出] 门柱检测结果
     * @param ball_id     足球类别 ID（1）
     * @param post_id     门柱类别 ID（0）
     * @param ball_thresh 足球置信度阈值
     * @param post_thresh 门柱置信度阈值
     * @param min_ball_w  足球最小宽度像素
     * @param min_ball_h  足球最小高度像素
     * @param min_post_w  门柱最小宽度像素
     * @param min_post_h  门柱最小高度像素
     * @param d_w_h       足球宽高比容差（|w/h - 1.0| < d_w_h）
     */
    void detect(unsigned char* dev_bgr, int img_w, int img_h,
                std::vector<object_det>& ball_dets,
                std::vector<object_det>& post_dets,
                int ball_id, int post_id,
                float ball_thresh, float post_thresh,
                int min_ball_w, int min_ball_h,
                int min_post_w, int min_post_h,
                float d_w_h);

    /**
     * @brief 释放所有 TensorRT 和 CUDA 资源
     * 替代原先的 free_network(net_) + cudaFree(dev_sized_) + cudaFree(dev_rgbfp_)
     */
    void release();

    /** @brief 获取网络输入宽度（用于日志/调试） */
    int net_w() const { return input_w_; }
    /** @brief 获取网络输入高度 */
    int net_h() const { return input_h_; }

private:
    static constexpr int kTensorNameMaxLen = 128;

    // ---- TensorRT 核心对象 ----
    nvinfer1::IRuntime*          runtime_ = nullptr;
    nvinfer1::ICudaEngine*       engine_  = nullptr;
    nvinfer1::IExecutionContext*  context_ = nullptr;

    // ---- Tensor名称（TensorRT 10 name-based API）----
    char input_name_[kTensorNameMaxLen] = {0};
    char output_name_[kTensorNameMaxLen] = {0};

    // ---- GPU 缓冲区 ----
    float* dev_input_  = nullptr;   // 网络输入 [1, 3, input_h_, input_w_]
    float* dev_output_ = nullptr;   // 网络输出 [1, 4+num_classes_, num_anchors_]

    // ---- CPU 缓冲区 ----
    float* host_output_ = nullptr;  // 输出拷贝到 CPU 进行后处理

    // ---- 网络参数 ----
    int input_w_      = 0;          // 网络输入宽度（如 640）
    int input_h_      = 0;          // 网络输入高度（如 640）
    int num_classes_   = 0;          // 类别数（2）
    int num_anchors_   = 0;          // 候选框数量（如 8400）
    int output_fields_ = 0;          // 每个 anchor 的字段数 (4 + num_classes_)
    int output_size_   = 0;          // 输出总 float 数

    // ---- letterbox 缩放参数（每次 detect 重新计算）----
    float scale_ = 1.0f;
    int   pad_x_ = 0;
    int   pad_y_ = 0;

    // ---- CUDA stream ----
    cudaStream_t stream_ = nullptr;

    // ---- 内部方法 ----

    /**
     * @brief Letterbox 预处理（GPU 端）
     * BGR uint8 HWC → RGB float32 CHW，保持宽高比，灰色填充
     */
    void preprocess(unsigned char* dev_bgr, int img_w, int img_h);

    /**
     * @brief YOLOv8 后处理
     * 解析输出张量 → 阈值过滤 → 坐标还原 → 尺寸过滤 → NMS
     */
    void postprocess(int img_w, int img_h,
                     std::vector<object_det>& ball_dets,
                     std::vector<object_det>& post_dets,
                     int ball_id, int post_id,
                     float ball_thresh, float post_thresh,
                     int min_ball_w, int min_ball_h,
                     int min_post_w, int min_post_h,
                     float d_w_h);

    /**
     * @brief CPU NMS（Non-Maximum Suppression）
     * 对已经按 prob 降序排列的 dets 做 IoU 抑制
     */
    static void nms(std::vector<object_det>& dets, float nms_thresh);

    /**
     * @brief 计算两个 object_det 的 IoU
     */
    static float iou(const object_det& a, const object_det& b);
};

#endif // __TRT_DETECTOR_HPP
