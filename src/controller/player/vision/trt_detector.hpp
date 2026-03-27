#ifndef __TRT_DETECTOR_HPP
#define __TRT_DETECTOR_HPP

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <NvInfer.h>

#include "model.hpp"

class TRTDetector
{
public:
    TRTDetector();
    ~TRTDetector();

    bool load(const std::string &engine_path, int ball_id, int post_id);
    void release();

    int input_w() const { return input_w_; }
    int input_h() const { return input_h_; }

    /**
     * @brief TensorRT 推理并填充目标检测结果（object_det）。
     *
     * 注意：
     * - 本项目的 Vision 侧预处理把图像拉伸到 detector 的输入分辨率，并产出 dev_rgbfp（RGB float CHW）。
     * - detect() 只负责：enqueueV3 -> 读取输出 -> YOLOv8 输出解析 -> 类别阈值/尺寸过滤 -> NMS -> 写回 ball_dets/post_dets
     */
    bool detect(float *dev_rgbfp,
                 int orig_w, int orig_h,
                 std::vector<object_det> &ball_dets,
                 std::vector<object_det> &post_dets,
                 float ball_thresh,
                 float post_thresh,
                 int min_ball_w,
                 int min_ball_h,
                 int min_post_w,
                 int min_post_h,
                 float d_w_h,
                 float letterbox_scale,
                 int letterbox_pad_x,
                 int letterbox_pad_y,
                 float nms_thresh = 0.45f);

private:
    bool parseOutputLayout(int num_dims,
                            const nvinfer1::Dims &dims,
                            int &output_fields,
                            int &num_anchors,
                            bool &fields_first) const;

    float readOutputValue(const std::vector<float> &host_out,
                          int anchor_idx,
                          int field_idx,
                          int output_fields,
                          int num_anchors,
                          bool fields_first) const;

    static float clampf(float v, float lo, float hi);

    static float iou(const object_det &a, const object_det &b);
    static void nms(std::vector<object_det> &dets, float nms_thresh);

private:
    class TRTLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char *msg) noexcept override
        {
            (void)severity;
            (void)msg;
        }
    };

    TRTLogger logger_;
    nvinfer1::IRuntime *runtime_;
    nvinfer1::ICudaEngine *engine_;
    nvinfer1::IExecutionContext *context_;

    std::string input_name_;
    std::string output_name_;

    int input_w_;
    int input_h_;

    nvinfer1::DataType output_dtype_;
    nvinfer1::DataType input_dtype_;
    size_t output_numel_;
    size_t output_elem_bytes_;

    void *dev_output_;
    cudaStream_t stream_;

    // class ids mapping
    int ball_id_;
    int post_id_;

    // Host output cache (always store as float for simpler parsing)
    std::vector<float> host_output_;
    std::vector<__half> host_output_half_;
};

#endif

