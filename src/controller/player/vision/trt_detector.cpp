#include "trt_detector.hpp"
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <sstream>
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
extern "C" cudaError_t cudaLetterboxPreprocess(
    const unsigned char* dev_bgr_src,
    float* dev_chw_dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale, int pad_x, int pad_y,
    cudaStream_t stream);

namespace {
bool is_dims_resolved(const nvinfer1::Dims& dims)
{
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0) return false;
    }
    return true;
}

bool check_cuda(const char* stage, cudaError_t err)
{
    if (err != cudaSuccess)
    {
        LOG(LOG_ERROR) << "TRTDetector: CUDA error at " << stage
                       << ": " << cudaGetErrorString(err) << endll;
        return false;
    }
    return true;
}

const char* io_mode_to_string(nvinfer1::TensorIOMode mode)
{
    switch (mode)
    {
    case nvinfer1::TensorIOMode::kINPUT:
        return "INPUT";
    case nvinfer1::TensorIOMode::kOUTPUT:
        return "OUTPUT";
    default:
        return "UNKNOWN";
    }
}

const char* dtype_to_string(nvinfer1::DataType dt)
{
    switch (dt)
    {
    case nvinfer1::DataType::kFLOAT:
        return "FP32";
    case nvinfer1::DataType::kHALF:
        return "FP16";
    case nvinfer1::DataType::kINT8:
        return "INT8";
    case nvinfer1::DataType::kINT32:
        return "INT32";
    case nvinfer1::DataType::kBOOL:
        return "BOOL";
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kUINT8:
        return "UINT8";
    case nvinfer1::DataType::kFP8:
        return "FP8";
    case nvinfer1::DataType::kBF16:
        return "BF16";
    case nvinfer1::DataType::kINT64:
        return "INT64";
#endif
    default:
        return "UNKNOWN";
    }
}

std::string dims_to_string(const nvinfer1::Dims& dims)
{
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (i > 0) oss << ", ";
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

size_t dims_volume(const nvinfer1::Dims& dims)
{
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0) return 0;
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}
} // namespace

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
    release();

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
    output_names_.clear();
    output_bytes_.clear();
    dev_outputs_.clear();
    primary_output_index_ = -1;
    int input_count = 0;
    int output_count = 0;
    for (int i = 0; i < nb_io_tensors; ++i)
    {
        const char* tensor_name = engine_->getIOTensorName(i);
        if (!tensor_name) continue;

        const auto mode = engine_->getTensorIOMode(tensor_name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            ++input_count;
            if (input_name_[0] == '\0')
            {
                std::strncpy(input_name_, tensor_name, kTensorNameMaxLen - 1);
                input_name_[kTensorNameMaxLen - 1] = '\0';
            }
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            ++output_count;
            output_names_.emplace_back(tensor_name);
        }
    }

    if (input_name_[0] == '\0' || output_names_.empty() || output_count < 1)
    {
        LOG(LOG_ERROR) << "TRTDetector: Failed to resolve input/output tensor names" << endll;
        return false;
    }
    if (input_count != 1)
    {
        LOG(LOG_ERROR) << "TRTDetector: Only single-input engines are supported, got inputs="
                       << input_count << endll;
        return false;
    }

    // 4. 获取输入维度 [1, 3, H, W]
    auto input_dims = engine_->getTensorShape(input_name_);
    if (input_dims.nbDims != 4)
    {
        LOG(LOG_ERROR) << "TRTDetector: Unexpected input dims: " << input_dims.nbDims << endll;
        return false;
    }

    // TensorRT 10: enqueueV3 前必须确保输入 shape 已明确
    nvinfer1::Dims4 fixed_dims{1, 3, 640, 640};
    if (input_dims.d[2] > 0 && input_dims.d[3] > 0)
    {
        fixed_dims = nvinfer1::Dims4{1, 3, input_dims.d[2], input_dims.d[3]};
    }
    if (!context_->setInputShape(input_name_, fixed_dims))
    {
        LOG(LOG_ERROR) << "TRTDetector: Failed to set input shape" << endll;
        return false;
    }

    input_dims = context_->getTensorShape(input_name_);
    if (!is_dims_resolved(input_dims))
    {
        LOG(LOG_ERROR) << "TRTDetector: Input shape unresolved after setInputShape" << endll;
        return false;
    }

    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];

    // 5. 打印 TRT10 运行时自检日志（每个 tensor 的 IO mode / dtype / shape）
    LOG(LOG_INFO) << "TRTDetector: TensorRT I/O self-check (after setInputShape)" << endll;
    for (int i = 0; i < nb_io_tensors; ++i)
    {
        const char* tensor_name = engine_->getIOTensorName(i);
        if (!tensor_name) continue;

        const auto mode = engine_->getTensorIOMode(tensor_name);
        const auto dtype = engine_->getTensorDataType(tensor_name);
        const auto shape = context_->getTensorShape(tensor_name);
        LOG(LOG_INFO) << "  [" << i << "] name=" << tensor_name
                      << " mode=" << io_mode_to_string(mode)
                      << " dtype=" << dtype_to_string(dtype)
                      << " shape=" << dims_to_string(shape) << endll;
    }

    // 6. 解析输出维度并选择主输出（用于 YOLO 后处理）
    bool found_primary = false;
    int best_anchor_count = -1;
    output_bytes_.resize(output_names_.size(), 0);
    dev_outputs_.resize(output_names_.size(), nullptr);

    for (size_t i = 0; i < output_names_.size(); ++i)
    {
        const char* out_name = output_names_[i].c_str();
        const auto out_dims = context_->getTensorShape(out_name);
        if (!is_dims_resolved(out_dims))
        {
            LOG(LOG_ERROR) << "TRTDetector: Output shape unresolved: " << out_name
                           << " shape=" << dims_to_string(out_dims) << endll;
            return false;
        }

        const size_t volume = dims_volume(out_dims);
        const int bytes_per_comp = engine_->getTensorBytesPerComponent(out_name);
        const int comp_per_elem = engine_->getTensorComponentsPerElement(out_name);
        const size_t out_bytes = volume * static_cast<size_t>(bytes_per_comp) * static_cast<size_t>(comp_per_elem);
        output_bytes_[i] = out_bytes;

        if (!check_cuda(("cudaMalloc(output:" + output_names_[i] + ")").c_str(),
                        cudaMalloc(&dev_outputs_[i], out_bytes)))
        {
            return false;
        }

        const auto dtype = engine_->getTensorDataType(out_name);
        if (dtype != nvinfer1::DataType::kFLOAT || out_dims.nbDims != 3 || comp_per_elem != 1)
        {
            continue;
        }

        // 兼容两种 YOLO 输出布局: [1, C, N] 或 [1, N, C]
        int fields = -1;
        int anchors = -1;
        bool anchor_major = false;
        if (out_dims.d[1] >= 5 && out_dims.d[1] <= 512)
        {
            fields = out_dims.d[1];
            anchors = out_dims.d[2];
            anchor_major = false;
        }
        else if (out_dims.d[2] >= 5 && out_dims.d[2] <= 512)
        {
            fields = out_dims.d[2];
            anchors = out_dims.d[1];
            anchor_major = true;
        }
        if (fields < 5 || anchors <= 0)
        {
            continue;
        }

        if (!found_primary || anchors > best_anchor_count)
        {
            found_primary = true;
            best_anchor_count = anchors;
            primary_output_index_ = static_cast<int>(i);
            num_anchors_ = anchors;
            output_fields_ = fields;
            output_anchor_major_ = anchor_major;
        }
    }

    if (!found_primary)
    {
        LOG(LOG_ERROR) << "TRTDetector: Cannot find a FP32 3D output suitable for YOLO postprocess" << endll;
        return false;
    }

    if (output_fields_ != (4 + num_classes_))
    {
        LOG(LOG_WARN) << "TRTDetector: Output fields=" << output_fields_
                      << " expected=" << (4 + num_classes_)
                      << ", adjusting num_classes" << endll;
        num_classes_ = output_fields_ - 4;
    }

    output_size_ = output_fields_ * num_anchors_;

    // 7. 分配 GPU 输入缓冲区
    if (!check_cuda("cudaMalloc(dev_input_)", cudaMalloc(&dev_input_, 3 * input_h_ * input_w_ * sizeof(float))))
    {
        return false;
    }

    // 8. 分配 CPU 主输出缓冲
    host_output_ = new float[output_size_];

    // 9. 创建 CUDA stream
    if (!check_cuda("cudaStreamCreate", cudaStreamCreate(&stream_)))
    {
        return false;
    }

    LOG(LOG_INFO) << "TRTDetector: Loaded engine " << engine_path
                  << " input=" << input_w_ << "x" << input_h_
                  << " classes=" << num_classes_
                  << " anchors=" << num_anchors_
                  << " outputs=" << output_names_.size()
                  << " primary_output=" << output_names_[primary_output_index_] << endll;

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
    if (!context_ || !dev_input_ || !host_output_ || primary_output_index_ < 0)
    {
        LOG(LOG_ERROR) << "TRTDetector: detect called before successful load" << endll;
        return;
    }

    // TensorRT 10: 每次推理前显式设置输入 shape，避免 profile/shape 状态漂移
    const nvinfer1::Dims4 run_dims{1, 3, input_h_, input_w_};
    if (!context_->setInputShape(input_name_, run_dims))
    {
        LOG(LOG_ERROR) << "TRTDetector: setInputShape failed at runtime" << endll;
        return;
    }

    // --- 阶段 1: 预处理 ---
    preprocess(dev_bgr, img_w, img_h);

    cudaError_t launch_err = cudaPeekAtLastError();
    if (!check_cuda("preprocess kernel", launch_err))
    {
        return;
    }

    // --- 阶段 2: TensorRT 推理 ---
    if (!context_->setTensorAddress(input_name_, dev_input_))
    {
        LOG(LOG_ERROR) << "TRTDetector: setTensorAddress failed for input" << endll;
        return;
    }
    for (size_t i = 0; i < output_names_.size(); ++i)
    {
        if (!context_->setTensorAddress(output_names_[i].c_str(), dev_outputs_[i]))
        {
            LOG(LOG_ERROR) << "TRTDetector: setTensorAddress failed for output: "
                           << output_names_[i] << endll;
            return;
        }
    }
    if (!context_->enqueueV3(stream_))
    {
        cudaError_t trt_cuda_err = cudaPeekAtLastError();
        if (trt_cuda_err != cudaSuccess)
        {
            LOG(LOG_ERROR) << "TRTDetector: CUDA state after enqueueV3 failure: "
                           << cudaGetErrorString(trt_cuda_err) << endll;
        }
        LOG(LOG_ERROR) << "TRTDetector: enqueueV3 failed" << endll;
        return;
    }

    // --- 阶段 3: 将输出拷贝到 CPU ---
    if (!check_cuda("cudaMemcpyAsync(D2H)",
                    cudaMemcpyAsync(host_output_, dev_outputs_[primary_output_index_],
                                    output_bytes_[primary_output_index_],
                                    cudaMemcpyDeviceToHost, stream_)))
    {
        return;
    }
    if (!check_cuda("cudaStreamSynchronize", cudaStreamSynchronize(stream_)))
    {
        return;
    }

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
    cudaError_t err = cudaLetterboxPreprocess(dev_bgr, dev_input_,
                                              img_w, img_h,
                                              input_w_, input_h_,
                                              scale_, pad_x_, pad_y_,
                                              stream_);
    if (err != cudaSuccess)
    {
        LOG(LOG_ERROR) << "TRTDetector: cudaLetterboxPreprocess failed: "
                       << cudaGetErrorString(err) << endll;
    }
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
        float cx, cy, bw, bh;
        float prob_post, prob_ball;

        if (!output_anchor_major_)
        {
            // 输出布局: [1, C, N]
            cx = host_output_[0 * num_anchors_ + i];
            cy = host_output_[1 * num_anchors_ + i];
            bw = host_output_[2 * num_anchors_ + i];
            bh = host_output_[3 * num_anchors_ + i];
            prob_post = host_output_[(4 + post_id) * num_anchors_ + i];
            prob_ball = host_output_[(4 + ball_id) * num_anchors_ + i];
        }
        else
        {
            // 输出布局: [1, N, C]
            const int base = i * output_fields_;
            cx = host_output_[base + 0];
            cy = host_output_[base + 1];
            bw = host_output_[base + 2];
            bh = host_output_[base + 3];
            prob_post = host_output_[base + (4 + post_id)];
            prob_ball = host_output_[base + (4 + ball_id)];
        }

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

    if (dev_input_) { cudaFree(dev_input_); dev_input_ = nullptr; }
    for (size_t i = 0; i < dev_outputs_.size(); ++i)
    {
        if (dev_outputs_[i])
        {
            cudaFree(dev_outputs_[i]);
            dev_outputs_[i] = nullptr;
        }
    }
    dev_outputs_.clear();
    output_bytes_.clear();
    output_names_.clear();
    primary_output_index_ = -1;

    if (host_output_) { delete[] host_output_; host_output_ = nullptr; }

    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}
