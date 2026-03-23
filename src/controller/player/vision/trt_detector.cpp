#include "trt_detector.hpp"

#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

using namespace std;

TRTDetector::TRTDetector()
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      input_w_(0),
      input_h_(0),
      output_dtype_(nvinfer1::DataType::kFLOAT),
      input_dtype_(nvinfer1::DataType::kFLOAT),
      output_numel_(0),
      output_elem_bytes_(0),
      dev_output_(nullptr),
      stream_(nullptr),
      ball_id_(1),
      post_id_(0)
{
}

TRTDetector::~TRTDetector()
{
    release();
}

bool TRTDetector::load(const std::string &engine_path, int ball_id, int post_id)
{
    ball_id_ = ball_id;
    post_id_ = post_id;

    // 1) read engine file
    std::ifstream file(engine_path.c_str(), std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        return false;
    }
    std::streamsize file_size = file.tellg();
    if (file_size <= 0)
    {
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(static_cast<size_t>(file_size));
    if (!file.read(engine_data.data(), file_size))
    {
        return false;
    }

    // 2) TRT10 runtime + engine
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_)
        return false;

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), static_cast<size_t>(file_size));
    if (!engine_)
        return false;

    context_ = engine_->createExecutionContext();
    if (!context_)
        return false;

    // 3) find input/output tensor names
    input_name_.clear();
    output_name_.clear();

    const int nb_tensors = engine_->getNbIOTensors();
    for (int i = 0; i < nb_tensors; ++i)
    {
        const char *name = engine_->getIOTensorName(i);
        if (!name)
            continue;
        const std::string sname(name);
        if (engine_->getTensorIOMode(sname.c_str()) == nvinfer1::TensorIOMode::kINPUT)
            input_name_ = sname;
        else if (engine_->getTensorIOMode(sname.c_str()) == nvinfer1::TensorIOMode::kOUTPUT)
            output_name_ = sname;
    }

    if (input_name_.empty() || output_name_.empty())
        return false;

    // 4) read shapes/dtypes
    auto in_dims = engine_->getTensorShape(input_name_.c_str());
    if (in_dims.nbDims != 4)
        return false;
    input_h_ = in_dims.d[2];
    input_w_ = in_dims.d[3];

    auto out_dims = engine_->getTensorShape(output_name_.c_str());
    if (out_dims.nbDims <= 0)
        return false;

    output_numel_ = 1;
    for (int i = 0; i < out_dims.nbDims; ++i)
        output_numel_ *= static_cast<size_t>(out_dims.d[i]);

    output_dtype_ = engine_->getTensorDataType(output_name_.c_str());
    input_dtype_ = engine_->getTensorDataType(input_name_.c_str());
    if (input_dtype_ != nvinfer1::DataType::kFLOAT)
        return false;
    if (!(output_dtype_ == nvinfer1::DataType::kFLOAT || output_dtype_ == nvinfer1::DataType::kHALF))
        return false;

    // 5) allocate output buffer (float)
    if (dev_output_)
    {
        cudaFree(dev_output_);
        dev_output_ = nullptr;
    }
    output_elem_bytes_ = (output_dtype_ == nvinfer1::DataType::kFLOAT) ? sizeof(float) : sizeof(__half);
    cudaError_t err = cudaMalloc(&dev_output_, output_numel_ * output_elem_bytes_);
    if (err != cudaSuccess)
        return false;

    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess)
        return false;

    host_output_.resize(output_numel_);
    if (output_dtype_ == nvinfer1::DataType::kHALF)
        host_output_half_.resize(output_numel_);
    else
        host_output_half_.clear();
    return true;
}

void TRTDetector::release()
{
    if (context_)
    {
        delete context_;
        context_ = nullptr;
    }
    if (engine_)
    {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_)
    {
        delete runtime_;
        runtime_ = nullptr;
    }

    if (dev_output_)
    {
        cudaFree(dev_output_);
        dev_output_ = nullptr;
    }
    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    host_output_.clear();
    host_output_half_.clear();
}

bool TRTDetector::parseOutputLayout(int num_dims,
                                    const nvinfer1::Dims &dims,
                                    int &output_fields,
                                    int &num_anchors,
                                    bool &fields_first) const
{
    (void)num_dims;
    if (dims.nbDims != 3)
        return false;
    if (dims.d[0] != 1)
        return false;

    // YOLOv8 exported for 2 classes: fields = 4 + nc = 6
    const int expected_fields = 6;

    const int d1 = dims.d[1];
    const int d2 = dims.d[2];
    if (d1 == expected_fields)
    {
        output_fields = d1;
        num_anchors = d2;
        fields_first = true;
        return true;
    }
    if (d2 == expected_fields)
    {
        output_fields = d2;
        num_anchors = d1;
        fields_first = false;
        return true;
    }
    return false;
}

float TRTDetector::readOutputValue(const std::vector<float> &host_out,
                                   int anchor_idx,
                                   int field_idx,
                                   int output_fields,
                                   int num_anchors,
                                   bool fields_first) const
{
    // fields_first: [fields, anchors]
    // anchors_first: [anchors, fields]
    if (fields_first)
    {
        const int idx = field_idx * num_anchors + anchor_idx;
        return host_out[static_cast<size_t>(idx)];
    }
    else
    {
        const int idx = anchor_idx * output_fields + field_idx;
        return host_out[static_cast<size_t>(idx)];
    }
}

float TRTDetector::iou(const object_det &a, const object_det &b)
{
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.w, b.x + b.w);
    const int y2 = std::min(a.y + a.h, b.y + b.h);

    const int inter_w = std::max(0, x2 - x1);
    const int inter_h = std::max(0, y2 - y1);
    const float inter_area = static_cast<float>(inter_w * inter_h);

    const float area_a = static_cast<float>(a.w * a.h);
    const float area_b = static_cast<float>(b.w * b.h);
    const float union_area = area_a + area_b - inter_area;
    if (union_area <= 0.0f)
        return 0.0f;
    return inter_area / union_area;
}

void TRTDetector::nms(std::vector<object_det> &dets, float nms_thresh)
{
    if (dets.empty())
        return;

    std::sort(dets.begin(), dets.end(),
              [](const object_det &a, const object_det &b) { return a.prob > b.prob; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<object_det> keep;
    keep.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i)
    {
        if (suppressed[i])
            continue;
        keep.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); ++j)
        {
            if (suppressed[j])
                continue;
            if (iou(dets[i], dets[j]) > nms_thresh)
                suppressed[j] = true;
        }
    }

    dets.swap(keep);
}

bool TRTDetector::detect(float *dev_rgbfp,
                          int orig_w,
                          int orig_h,
                          std::vector<object_det> &ball_dets,
                          std::vector<object_det> &post_dets,
                          float ball_thresh,
                          float post_thresh,
                          int min_ball_w,
                          int min_ball_h,
                          int min_post_w,
                          int min_post_h,
                          float d_w_h,
                          float nms_thresh)
{
    if (!engine_ || !context_ || !dev_output_ || !dev_rgbfp)
        return false;

    // 1) enqueue
    context_->setTensorAddress(input_name_.c_str(), dev_rgbfp);
    context_->setTensorAddress(output_name_.c_str(), dev_output_);

    if (!context_->enqueueV3(stream_))
        return false;

    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess)
        return false;

    if (output_dtype_ == nvinfer1::DataType::kFLOAT)
    {
        err = cudaMemcpy(host_output_.data(),
                         dev_output_,
                         output_numel_ * sizeof(float),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            return false;
    }
    else
    {
        err = cudaMemcpy(host_output_half_.data(),
                         dev_output_,
                         output_numel_ * sizeof(__half),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            return false;

        for (size_t i = 0; i < output_numel_; ++i)
            host_output_[i] = __half2float(host_output_half_[i]);
    }

    // 2) parse output layout
    int output_fields = 0;
    int num_anchors = 0;
    bool fields_first = true;

    auto out_dims = engine_->getTensorShape(output_name_.c_str());
    if (!parseOutputLayout(out_dims.nbDims, out_dims, output_fields, num_anchors, fields_first))
        return false;

    // 3) heuristic: detect if coords are normalized
    float max_abs = 0.0f;
    const int sample = std::min(num_anchors, 50);
    for (int i = 0; i < sample; ++i)
    {
        const float cx = readOutputValue(host_output_, i, 0, output_fields, num_anchors, fields_first);
        const float cy = readOutputValue(host_output_, i, 1, output_fields, num_anchors, fields_first);
        const float bw = readOutputValue(host_output_, i, 2, output_fields, num_anchors, fields_first);
        const float bh = readOutputValue(host_output_, i, 3, output_fields, num_anchors, fields_first);
        max_abs = std::max(max_abs, std::max(std::fabs(cx), std::max(std::fabs(cy), std::max(std::fabs(bw), std::fabs(bh)))));
    }
    const bool coords_normalized = (max_abs <= 2.0f);

    // 4) decode anchors
    ball_dets.clear();
    post_dets.clear();

    const float scale_x = static_cast<float>(orig_w) / static_cast<float>(input_w_);
    const float scale_y = static_cast<float>(orig_h) / static_cast<float>(input_h_);

    for (int i = 0; i < num_anchors; ++i)
    {
        float cx = readOutputValue(host_output_, i, 0, output_fields, num_anchors, fields_first);
        float cy = readOutputValue(host_output_, i, 1, output_fields, num_anchors, fields_first);
        float bw = readOutputValue(host_output_, i, 2, output_fields, num_anchors, fields_first);
        float bh = readOutputValue(host_output_, i, 3, output_fields, num_anchors, fields_first);

        float prob_post = readOutputValue(host_output_, i, 4 + post_id_, output_fields, num_anchors, fields_first);
        float prob_ball = readOutputValue(host_output_, i, 4 + ball_id_, output_fields, num_anchors, fields_first);

        if (coords_normalized)
        {
            cx *= static_cast<float>(input_w_);
            cy *= static_cast<float>(input_h_);
            bw *= static_cast<float>(input_w_);
            bh *= static_cast<float>(input_h_);
        }

        // pick class by max prob (matches darknet parsing behavior)
        const bool is_ball = (prob_ball > prob_post);

        // convert from center to left-top in original image pixel space
        const float real_cx = cx * scale_x;
        const float real_cy = cy * scale_y;
        const float real_w = bw * scale_x;
        const float real_h = bh * scale_y;

        const float x_left = real_cx - real_w * 0.5f;
        const float y_top = real_cy - real_h * 0.5f;

        if (is_ball)
        {
            if (prob_ball < ball_thresh)
                continue;

            if (real_w < static_cast<float>(min_ball_w) || real_h < static_cast<float>(min_ball_h))
                continue;

            const float ratio = real_w / std::max(1.0f, real_h);
            if (std::fabs(ratio - 1.0f) >= d_w_h)
                continue;

            ball_dets.push_back(object_det(ball_id_, prob_ball,
                                            static_cast<int>(x_left),
                                            static_cast<int>(y_top),
                                            static_cast<int>(real_w),
                                            static_cast<int>(real_h) + 1));
        }
        else
        {
            if (prob_post < post_thresh)
                continue;

            if (real_w < static_cast<float>(min_post_w) || real_h < static_cast<float>(min_post_h))
                continue;

            post_dets.push_back(object_det(post_id_, prob_post,
                                            static_cast<int>(x_left),
                                            static_cast<int>(y_top),
                                            static_cast<int>(real_w),
                                            static_cast<int>(real_h)));
        }
    }

    // 5) class-wise NMS + sort
    nms(ball_dets, nms_thresh);
    nms(post_dets, nms_thresh);

    // Keep same descending order as darknet (Vision::run used rbegin/rend)
    std::sort(ball_dets.begin(), ball_dets.end(),
              [](const object_det &a, const object_det &b) { return a.prob > b.prob; });
    std::sort(post_dets.begin(), post_dets.end(),
              [](const object_det &a, const object_det &b) { return a.prob > b.prob; });

    return true;
}

