#include "trt_detector.hpp"
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include "logger.hpp"  // 项目日志宏 LOG()

// ============================================================
// TensorRT Logger（必须提供，TRT 通过它输出警告/错误信息）
// ============================================================
class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // 只输出警告及以上级别
        if (severity <= Severity::kWARNING)
        {
            LOG(LOG_WARN) << "[TensorRT] " << msg << endll;
        }
    }
};
static TRTLogger gLogger;

// ============================================================
// 前向声明 CUDA kernel（在 trt_preprocess.cu 中实现）
// ============================================================
extern "C" void cudaLetterboxPreprocess(
    const unsigned char* dev_bgr_src,
    float* dev_chw_dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale, int pad_x, int pad_y,
    cudaStream_t stream);

// ============================================================
// 构造 / 析构
// ============================================================
TRTDetector::TRTDetector()  {}
TRTDetector::~TRTDetector() { release(); }

// ============================================================
// load() — 加载 TensorRT Engine
// ============================================================
bool TRTDetector::load(const std::string& engine_path, int num_classes)
{
    num_classes_ = num_classes;
    output_fields_ = 4 + num_classes_;

    // 1. 读取 engine 文件到内存
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        LOG(LOG_ERROR) << "TRTDetector: Cannot open engine file: " << engine_path << endll;
        return false;
    }
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(file_size);
    file.read(engine_data.data(), file_size);
    file.close();

    // 2. 创建 TensorRT Runtime + 反序列化 Engine
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_)
    {
        LOG(LOG_ERROR) << "TRTDetector: Failed to create TensorRT runtime" << endll;
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), file_size);
    if (!engine_)
    {
        LOG(LOG_ERROR) << "TRTDetector: Failed to deserialize engine" << endll;
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_)
    {
        LOG(LOG_ERROR) << "TRTDetector: Failed to create execution context" << endll;
        return false;
    }

    // 3. 枚举 IO Tensor，获取输入输出名称（TensorRT 10）
    const int nb_io_tensors = engine_->getNbIOTensors();
    if (nb_io_tensors < 2)
    {
        LOG(LOG_ERROR) << "TRTDetector: Unexpected IO tensor count: " << nb_io_tensors << endll;
        return false;
    }

    input_name_[0] = '\0';
    output_name_[0] = '\0';
    for (int i = 0; i < nb_io_tensors; ++i)
    {
        const char* tensor_name = engine_->getIOTensorName(i);
        if (!tensor_name) continue;

        const auto mode = engine_->getTensorIOMode(tensor_name);
        if (mode == nvinfer1::TensorIOMode::kINPUT && input_name_[0] == '\0')
        {
            std::strncpy(input_name_, tensor_name, kTensorNameMaxLen - 1);
            input_name_[kTensorNameMaxLen - 1] = '\0';
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT && output_name_[0] == '\0')
        {
            std::strncpy(output_name_, tensor_name, kTensorNameMaxLen - 1);
            output_name_[kTensorNameMaxLen - 1] = '\0';
        }
    }

    if (input_name_[0] == '\0' || output_name_[0] == '\0')
    {
        LOG(LOG_ERROR) << "TRTDetector: Failed to resolve input/output tensor names" << endll;
        return false;
    }

    // 4. 获取输入维度 [1, 3, H, W]
    auto input_dims = engine_->getTensorShape(input_name_);
    if (input_dims.nbDims != 4)
    {
        LOG(LOG_ERROR) << "TRTDetector: Unexpected input dims: " << input_dims.nbDims << endll;
        return false;
    }

    // 动态输入时，先设置 shape 再查询输出 shape
    if (input_dims.d[0] < 0 || input_dims.d[1] < 0 || input_dims.d[2] < 0 || input_dims.d[3] < 0)
    {
        nvinfer1::Dims4 fixed_dims{1, 3, 640, 640};
        if (!context_->setInputShape(input_name_, fixed_dims))
        {
            LOG(LOG_ERROR) << "TRTDetector: Failed to set dynamic input shape" << endll;
            return false;
        }
        input_dims = context_->getTensorShape(input_name_);
    }

    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];

    // 5. 获取输出维度 [1, 4+nc, num_anchors]
    auto output_dims = context_->getTensorShape(output_name_);
    if (output_dims.nbDims == 3)
    {
        // 标准 YOLOv8 输出: [1, 4+nc, 8400]
        num_anchors_ = output_dims.d[2];
        int fields = output_dims.d[1];
        if (fields != output_fields_)
        {
            LOG(LOG_WARN) << "TRTDetector: Output fields=" << fields
                          << " expected=" << output_fields_
                          << ", adjusting num_classes" << endll;
            num_classes_ = fields - 4;
            output_fields_ = fields;
        }
    }
    else
    {
        LOG(LOG_ERROR) << "TRTDetector: Unexpected output dims: " << output_dims.nbDims << endll;
        return false;
    }

    output_size_ = output_fields_ * num_anchors_;

    // 6. 分配 GPU 缓冲区
    cudaMalloc(&dev_input_,  3 * input_h_ * input_w_ * sizeof(float));
    cudaMalloc(&dev_output_, output_size_ * sizeof(float));

    // 7. 分配 CPU 输出缓冲
    host_output_ = new float[output_size_];

    // 8. 创建 CUDA stream
    cudaStreamCreate(&stream_);

    LOG(LOG_INFO) << "TRTDetector: Loaded engine " << engine_path
                  << " input=" << input_w_ << "x" << input_h_
                  << " classes=" << num_classes_
                  << " anchors=" << num_anchors_ << endll;

    return true;
}

// ============================================================
// detect() — 完整检测流程
// ============================================================
void TRTDetector::detect(unsigned char* dev_bgr, int img_w, int img_h,
                          std::vector<object_det>& ball_dets,
                          std::vector<object_det>& post_dets,
                          int ball_id, int post_id,
                          float ball_thresh, float post_thresh,
                          int min_ball_w, int min_ball_h,
                          int min_post_w, int min_post_h,
                          float d_w_h)
{
    // --- 阶段 1: 预处理 ---
    preprocess(dev_bgr, img_w, img_h);

    // --- 阶段 2: TensorRT 推理 ---
    if (!context_->setTensorAddress(input_name_, dev_input_))
    {
        LOG(LOG_ERROR) << "TRTDetector: setTensorAddress failed for input" << endll;
        return;
    }
    if (!context_->setTensorAddress(output_name_, dev_output_))
    {
        LOG(LOG_ERROR) << "TRTDetector: setTensorAddress failed for output" << endll;
        return;
    }
    if (!context_->enqueueV3(stream_))
    {
        LOG(LOG_ERROR) << "TRTDetector: enqueueV3 failed" << endll;
        return;
    }

    // --- 阶段 3: 将输出拷贝到 CPU ---
    cudaMemcpyAsync(host_output_, dev_output_,
                    output_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // --- 阶段 4: 后处理 ---
    postprocess(img_w, img_h, ball_dets, post_dets,
                ball_id, post_id, ball_thresh, post_thresh,
                min_ball_w, min_ball_h, min_post_w, min_post_h, d_w_h);
}

// ============================================================
// preprocess() — Letterbox + BGR→RGB + 归一化 + HWC→CHW
// ============================================================
void TRTDetector::preprocess(unsigned char* dev_bgr, int img_w, int img_h)
{
    // 计算 letterbox 参数
    float scale_w = (float)input_w_ / img_w;
    float scale_h = (float)input_h_ / img_h;
    scale_ = std::min(scale_w, scale_h);

    int new_w = (int)(img_w * scale_);
    int new_h = (int)(img_h * scale_);
    pad_x_ = (input_w_ - new_w) / 2;
    pad_y_ = (input_h_ - new_h) / 2;

    // 先用灰色（114/255）填充整个输入缓冲
    // 然后在 kernel 中写入 resize 后的有效区域
    // 全部在 GPU 端完成
    cudaLetterboxPreprocess(dev_bgr, dev_input_,
                            img_w, img_h,
                            input_w_, input_h_,
                            scale_, pad_x_, pad_y_,
                            stream_);
}

// ============================================================
// postprocess() — YOLOv8 输出解析
// ============================================================
void TRTDetector::postprocess(int img_w, int img_h,
                               std::vector<object_det>& ball_dets,
                               std::vector<object_det>& post_dets,
                               int ball_id, int post_id,
                               float ball_thresh, float post_thresh,
                               int min_ball_w, int min_ball_h,
                               int min_post_w, int min_post_h,
                               float d_w_h)
{
    ball_dets.clear();
    post_dets.clear();

    /*
     * YOLOv8 输出张量: [1, output_fields_, num_anchors_]
     * 其中 output_fields_ = 4 + num_classes_ = 6
     *
     * 数据排列（列优先）:
     *   host_output_[0 * num_anchors_ + i] = cx  (第 i 个 anchor 的中心 x)
     *   host_output_[1 * num_anchors_ + i] = cy
     *   host_output_[2 * num_anchors_ + i] = w
     *   host_output_[3 * num_anchors_ + i] = h
     *   host_output_[4 * num_anchors_ + i] = class_0_prob (post)
     *   host_output_[5 * num_anchors_ + i] = class_1_prob (ball)
     *
     * 坐标是相对于网络输入尺寸 (640×640) 的像素值
     * 注意：没有 objectness score，直接取类别概率
     */

    for (int i = 0; i < num_anchors_; i++)
    {
        // 读取中心坐标和宽高（相对于 640×640 letterbox 图像）
        float cx = host_output_[0 * num_anchors_ + i];
        float cy = host_output_[1 * num_anchors_ + i];
        float bw = host_output_[2 * num_anchors_ + i];
        float bh = host_output_[3 * num_anchors_ + i];

        // 读取各类别概率
        float prob_post = host_output_[(4 + post_id) * num_anchors_ + i];
        float prob_ball = host_output_[(4 + ball_id) * num_anchors_ + i];

        // ---- 坐标从 letterbox 空间还原到原图像素空间 ----
        float real_cx = (cx - pad_x_) / scale_;
        float real_cy = (cy - pad_y_) / scale_;
        float real_w  = bw / scale_;
        float real_h  = bh / scale_;

        // 转为左上角坐标（与原 object_det 格式一致）
        int x = (int)(real_cx - real_w / 2.0f);
        int y = (int)(real_cy - real_h / 2.0f);
        int w = (int)real_w;
        int h = (int)real_h;

        // ---- 分类分发（与原 Darknet 解析逻辑完全一致）----
        if (prob_ball > prob_post)
        {
            // 球
            if (prob_ball >= ball_thresh)
            {
                float w_h_ratio = (float)w / (float)(h > 0 ? h : 1);
                if (w >= min_ball_w && h >= min_ball_h && fabs(w_h_ratio - 1.0f) < d_w_h)
                {
                    ball_dets.push_back(object_det(ball_id, prob_ball, x, y, w, h));
                }
            }
        }
        else
        {
            // 门柱
            if (prob_post >= post_thresh)
            {
                if (w >= min_post_w && h >= min_post_h)
                {
                    post_dets.push_back(object_det(post_id, prob_post, x, y, w, h));
                }
            }
        }
    }

    // NMS（分别对球和门柱做）
    nms(ball_dets, 0.45f);
    nms(post_dets, 0.45f);

    // 按置信度降序排列（与原代码 rbegin/rend 排序一致）
    std::sort(ball_dets.rbegin(), ball_dets.rend());
    std::sort(post_dets.rbegin(), post_dets.rend());
}

// ============================================================
// nms() — 非极大值抑制
// ============================================================
float TRTDetector::iou(const object_det& a, const object_det& b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.w, b.x + b.w);
    int y2 = std::min(a.y + a.h, b.y + b.h);

    int inter_w = std::max(0, x2 - x1);
    int inter_h = std::max(0, y2 - y1);
    float inter_area = (float)(inter_w * inter_h);

    float area_a = (float)(a.w * a.h);
    float area_b = (float)(b.w * b.h);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

void TRTDetector::nms(std::vector<object_det>& dets, float nms_thresh)
{
    // 先按 prob 降序排列
    std::sort(dets.rbegin(), dets.rend());

    std::vector<object_det> result;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); i++)
    {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); j++)
        {
            if (suppressed[j]) continue;
            if (iou(dets[i], dets[j]) > nms_thresh)
            {
                suppressed[j] = true;
            }
        }
    }

    dets = std::move(result);
}

// ============================================================
// release() — 释放所有资源
// ============================================================
void TRTDetector::release()
{
    if (context_) { delete context_; context_ = nullptr; }
    if (engine_)  { delete engine_;  engine_  = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }

    if (dev_input_)   { cudaFree(dev_input_);   dev_input_  = nullptr; }
    if (dev_output_)  { cudaFree(dev_output_);  dev_output_ = nullptr; }
    if (host_output_) { delete[] host_output_;   host_output_ = nullptr; }

    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}
