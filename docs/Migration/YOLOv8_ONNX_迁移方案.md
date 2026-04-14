# YOLOv3 Darknet → YOLOv8 ONNX 迁移方案

> **平台**：NVIDIA Jetson Orin NX (aarch64)  
> **目标**：将视觉检测器从 YOLOv3 Darknet C API 替换为 YOLOv8 ONNX（TensorRT 加速），保持原代码架构和下游模块不变  
> **日期**：2026-02-25

---

## 一、当前架构分析

### 1.1 Darknet 调用链（需要替换的部分）

| 阶段 | 当前 API | 所在函数 | 文件位置 |
|------|---------|---------|---------|
| 加载模型 | `parse_network_cfg_custom()` + `load_weights()` | `start()` | vision.cpp L489-490 |
| 优化 | `set_batch_network()` / `fuse_conv_batchnorm()` / `calculate_binary_weights()` | `start()` | vision.cpp L492-494 |
| 预处理 | `cudaResizePacked()` + `cudaBGR2RGBfp()` | `run()` | vision.cpp L189-190 |
| 推理 | `network_predict(net_, dev_rgbfp_, 0)` | `run()` | vision.cpp L198 |
| 后处理 | `get_network_boxes()` + `do_nms_sort()` | `run()` | vision.cpp L201-204 |
| 释放检测 | `free_detections()` | `run()` | vision.cpp L239 |
| 释放网络 | `free_network()` | `stop()` | vision.cpp L546 |

### 1.2 数据接口（不需要改动）

检测结果最终写入 `std::vector<object_det> ball_dets_` 和 `post_dets_`：

```cpp
struct object_det {
    int id;              // 类别ID (0=门柱, 1=足球)
    float prob;          // 置信度
    int x, y, w, h;     // 像素坐标 (左上角+宽高)
};
```

下游所有模块（`odometry`、`camera2self`、世界模型更新、定位、调试可视化）**只依赖 `ball_dets_` / `post_dets_`**，与检测器实现完全解耦。

### 1.3 受影响的文件清单

| 文件 | 改动程度 | 说明 |
|------|---------|------|
| `src/controller/player/vision/vision.hpp` | **大** | 替换 `network net_` 成员为推理引擎 |
| `src/controller/player/vision/vision.cpp` | **大** | `start()`/`run()`/`stop()` 核心改动 |
| `src/controller/player/vision/CMakeLists.txt` | **中** | 链接库替换 |
| `src/data/config.conf` | **小** | 模型路径配置项变更 |
| `src/lib/imageproc/imageproc.hpp` | **小** | 移除不必要的 `#include "darknet/network.h"` |
| `src/lib/imageproc/color/color.cpp` | **小** | `check_error()` 依赖 `darknet/cuda.h` |
| `src/tools/auto_marker/main.cpp` | **中** | 独立工具，需同步迁移或保留双版本 |
| `src/lib/CMakeLists.txt` | **可选** | 是否保留 darknet 子目录 |

### 1.4 不受影响的模块

- `odometry()` 测距逻辑
- `camera2self()` 坐标变换
- 世界模型 (`WorldModel`)
- 自定位 (`SelfLocalization`)
- 调试图像发送 (`send_image`)
- 所有通信协议

---

## 二、推理后端选择

### 2.1 方案对比

| 方案 | 推理速度 | 集成复杂度 | Orin NX 支持 | 推荐度 |
|------|---------|-----------|-------------|--------|
| **A. TensorRT (推荐)** | ★★★★★ | 中 | 原生最优 | ⭐ **首选** |
| B. ONNX Runtime + CUDA EP | ★★★★ | 低 | 良好 | 备选 |
| C. ONNX Runtime + TensorRT EP | ★★★★★ | 中 | 良好 | 备选 |

**推荐方案 A：直接使用 TensorRT C++ API**

理由：
1. Orin NX 上 TensorRT 已预装（JetPack 自带）
2. 推理延迟最低（INT8/FP16 加速）
3. 输入输出直接操作 GPU 内存，与现有 CUDA Pipeline 无缝衔接
4. 避免额外安装 ONNX Runtime

### 2.2 推理流程对比

```
【当前 Darknet】
dev_rgbfp_ (GPU float*) → network_predict() → get_network_boxes() → object_det

【方案 A: TensorRT】
dev_input_ (GPU float*) → context->enqueueV2() → 自定义后处理 → object_det

【方案 B: ONNX Runtime】
host_input (CPU float*) → session.Run() → 自定义后处理 → object_det
```

---

## 三、模型转换步骤

### 3.1 训练 YOLOv8 模型

```bash
# 安装 ultralytics
pip install ultralytics

# 使用现有标注数据训练（YOLO格式）
yolo detect train data=robocup.yaml model=yolov8n.pt epochs=200 imgsz=640
```

`robocup.yaml` 内容（与原 Darknet 类别对应）：
```yaml
names:
  0: post    # 对应原 post_id_ = 0
  1: ball    # 对应原 ball_id_ = 1
nc: 2
# 数据集路径
train: ./dataset/train/images
val: ./dataset/val/images
```

> **注意**：确保类别 ID 与原代码中 `ball_id_=1`、`post_id_=0` 一致！

### 3.2 导出 ONNX

```bash
yolo export model=best.pt format=onnx imgsz=640 opset=11 simplify=True
```

生成 `best.onnx`。

### 3.3 转换为 TensorRT Engine（在 Orin NX 上执行）

```bash
# FP16 精度（推荐，速度与精度的平衡）
/usr/src/tensorrt/bin/trtexec \
    --onnx=best.onnx \
    --saveEngine=best.engine \
    --fp16 \
    --workspace=1024

# 或 INT8 精度（需要校准数据集，速度更快）
/usr/src/tensorrt/bin/trtexec \
    --onnx=best.onnx \
    --saveEngine=best_int8.engine \
    --int8 \
    --calib=calibration_cache.txt \
    --workspace=1024
```

> **关键**：TensorRT Engine 与 GPU 架构绑定，**必须在目标 Orin NX 设备上生成**，不可跨平台移植。

### 3.4 验证输入输出维度

```bash
# 查看 ONNX 模型信息
python -c "
import onnx
model = onnx.load('best.onnx')
for inp in model.graph.input:
    print('Input:', inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim])
for out in model.graph.output:
    print('Output:', out.name, [d.dim_value for d in out.type.tensor_type.shape.dim])
"
```

典型 YOLOv8 输出：
- **输入**：`images` → `[1, 3, 640, 640]` (NCHW, float32, 归一化到 [0,1])
- **输出**：`output0` → `[1, 84, 8400]` (84 = 4坐标 + 80类，这里是2类则为 `[1, 6, 8400]`)

> **注意**：YOLOv8 输出格式与 YOLOv3 不同！YOLOv8 没有 objectness score，直接输出类别概率。

---

## 四、代码改动详细步骤

### 步骤 1：创建 TensorRT 推理封装类

创建新文件 `src/controller/player/vision/trt_detector.hpp` 和 `.cpp`，封装 TensorRT 推理逻辑：

```cpp
// trt_detector.hpp
#ifndef __TRT_DETECTOR_HPP
#define __TRT_DETECTOR_HPP

#include <string>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include "model.hpp"    // object_det

class TRTDetector
{
public:
    TRTDetector();
    ~TRTDetector();

    // 加载 .engine 或 .onnx 文件
    bool load(const std::string& engine_path, int num_classes);

    // 推理：输入 GPU 上的 BGR unsigned char* 图像
    // 输出填充到 ball_dets 和 post_dets
    void detect(unsigned char* dev_bgr, int img_w, int img_h,
                std::vector<object_det>& ball_dets,
                std::vector<object_det>& post_dets,
                int ball_id, int post_id,
                float ball_thresh, float post_thresh,
                int min_ball_w, int min_ball_h,
                int min_post_w, int min_post_h,
                float d_w_h);

    void release();

    int net_w() const { return input_w_; }
    int net_h() const { return input_h_; }

private:
    // TensorRT 核心对象
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // GPU 缓冲区
    float* dev_input_ = nullptr;    // [1, 3, H, W]
    float* dev_output_ = nullptr;   // [1, num_det_fields, num_anchors]
    float* host_output_ = nullptr;  // CPU 端输出

    int input_w_, input_h_;
    int num_classes_;
    int num_anchors_;
    int output_size_;

    // 预处理：BGR uint8 → RGB float32 归一化 + letterbox
    void preprocess(unsigned char* dev_bgr, int img_w, int img_h);

    // 后处理：解析输出 → object_det
    void postprocess(int img_w, int img_h,
                     std::vector<object_det>& ball_dets,
                     std::vector<object_det>& post_dets,
                     int ball_id, int post_id,
                     float ball_thresh, float post_thresh,
                     int min_ball_w, int min_ball_h,
                     int min_post_w, int min_post_h,
                     float d_w_h);

    // NMS
    void nms(std::vector<object_det>& dets, float nms_thresh);

    // letterbox 缩放参数
    float scale_;
    int pad_x_, pad_y_;
};

#endif
```

### 步骤 2：实现 TRTDetector

`src/controller/player/vision/trt_detector.cpp` 核心实现要点：

```cpp
#include "trt_detector.hpp"
#include <fstream>
#include <algorithm>
#include <cassert>
#include "imageproc/imageproc.hpp"  // 复用已有 CUDA 函数

// TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};
static TRTLogger gLogger;

bool TRTDetector::load(const std::string& engine_path, int num_classes)
{
    num_classes_ = num_classes;

    // 读取 engine 文件
    std::ifstream file(engine_path, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    context_ = engine_->createExecutionContext();

    // 获取输入输出维度
    auto input_dims = engine_->getBindingDimensions(0);  // [1,3,H,W]
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];

    auto output_dims = engine_->getBindingDimensions(1); // [1, 4+nc, num_anchors]
    num_anchors_ = output_dims.d[2];
    output_size_ = (4 + num_classes_) * num_anchors_;

    // 分配 GPU 缓冲区
    cudaMalloc(&dev_input_, 3 * input_h_ * input_w_ * sizeof(float));
    cudaMalloc(&dev_output_, output_size_ * sizeof(float));
    host_output_ = new float[output_size_];

    return true;
}

void TRTDetector::detect(unsigned char* dev_bgr, int img_w, int img_h,
                          std::vector<object_det>& ball_dets,
                          std::vector<object_det>& post_dets,
                          int ball_id, int post_id,
                          float ball_thresh, float post_thresh,
                          int min_ball_w, int min_ball_h,
                          int min_post_w, int min_post_h,
                          float d_w_h)
{
    preprocess(dev_bgr, img_w, img_h);

    // TensorRT 推理
    void* bindings[] = { dev_input_, dev_output_ };
    context_->enqueueV2(bindings, 0, nullptr);

    // 拷贝输出到 CPU
    cudaMemcpy(host_output_, dev_output_,
               output_size_ * sizeof(float), cudaMemcpyDeviceToHost);

    postprocess(img_w, img_h, ball_dets, post_dets,
                ball_id, post_id, ball_thresh, post_thresh,
                min_ball_w, min_ball_h, min_post_w, min_post_h, d_w_h);
}
```

#### 预处理关键差异

```cpp
void TRTDetector::preprocess(unsigned char* dev_bgr, int img_w, int img_h)
{
    // YOLOv8 需要 letterbox (保持宽高比) 而非 Darknet 的直接 resize
    // 1. 计算缩放比例
    scale_ = std::min((float)input_w_ / img_w, (float)input_h_ / img_h);
    int new_w = (int)(img_w * scale_);
    int new_h = (int)(img_h * scale_);
    pad_x_ = (input_w_ - new_w) / 2;
    pad_y_ = (input_h_ - new_h) / 2;

    // 2. GPU 上执行 resize + pad + BGR→RGB + 归一化到[0,1]
    // 可编写 CUDA kernel 或组合现有函数实现
    // cudaLetterboxPreprocess(dev_bgr, dev_input_, img_w, img_h,
    //                         input_w_, input_h_, scale_, pad_x_, pad_y_);
}
```

#### 后处理关键差异

```cpp
void TRTDetector::postprocess(int img_w, int img_h, ...)
{
    ball_dets.clear();
    post_dets.clear();

    // YOLOv8 输出: [1, 4+nc, 8400] → 转置为 [8400, 4+nc]
    // 每个 anchor: [cx, cy, w, h, class0_prob, class1_prob, ...]
    // 注意：YOLOv8 没有 objectness score，直接取类别最大概率

    for (int i = 0; i < num_anchors_; i++)
    {
        float cx = host_output_[0 * num_anchors_ + i];
        float cy = host_output_[1 * num_anchors_ + i];
        float w  = host_output_[2 * num_anchors_ + i];
        float h  = host_output_[3 * num_anchors_ + i];

        // 取各类别概率
        float prob_post = host_output_[(4 + post_id) * num_anchors_ + i];
        float prob_ball = host_output_[(4 + ball_id) * num_anchors_ + i];

        // 还原 letterbox → 原图坐标
        cx = (cx - pad_x_) / scale_;
        cy = (cy - pad_y_) / scale_;
        w  = w / scale_;
        h  = h / scale_;

        int bx = (int)(cx - w / 2.0f);
        int by = (int)(cy - h / 2.0f);
        int bw = (int)w;
        int bh = (int)h;

        if (prob_ball > prob_post)
        {
            if (prob_ball >= ball_thresh)
            {
                float w_h = (float)bw / std::max(1, bh);
                if (bw >= min_ball_w && bh >= min_ball_h && fabs(w_h - 1.0f) < d_w_h)
                    ball_dets.push_back(object_det(ball_id, prob_ball, bx, by, bw, bh));
            }
        }
        else
        {
            if (prob_post >= post_thresh)
            {
                if (bw >= min_post_w && bh >= min_post_h)
                    post_dets.push_back(object_det(post_id, prob_post, bx, by, bw, bh));
            }
        }
    }

    // NMS
    nms(ball_dets, 0.45f);
    nms(post_dets, 0.45f);

    std::sort(ball_dets.rbegin(), ball_dets.rend());
    std::sort(post_dets.rbegin(), post_dets.rend());
}
```

### 步骤 3：修改 vision.hpp

```diff
 #ifndef __VISION_HPP
 #define __VISION_HPP

 // ... 其他 include 保持不变 ...
-#include "darknet/network.h"
+#include "trt_detector.hpp"

 class Vision: public Timer, public Subscriber, public Singleton<Vision>
 {
     // ... 公有接口完全不变 ...

 private:
-    network net_;
+    TRTDetector detector_;

     // 以下成员可移除（被 TRTDetector 内部管理）：
-    unsigned char *dev_sized_;
-    float *dev_rgbfp_;
-    int sized_size_;
-    int rgbf_size_;

     // 其余所有成员保持不变
 };
```

### 步骤 4：修改 vision.cpp

#### 4.1 移除 Darknet 头文件

```diff
 #include "vision.hpp"
 #include "parser/parser.hpp"
-#include "darknet/parser.h"
 #include <cuda_runtime.h>
```

#### 4.2 修改 start()

```diff
 bool Vision::start()
 {
-    net_.gpu_index = 0;
-    net_ = parse_network_cfg_custom(...);
-    load_weights(&net_, ...);
-    set_batch_network(&net_, 1);
-    fuse_conv_batchnorm(net_);
-    calculate_binary_weights(net_);
-    srand(2222222);
+    std::string engine_path = CONF->get_config_value<std::string>("net_engine_file");
+    if (!detector_.load(engine_path, 2))  // 2个类别
+    {
+        LOG(LOG_ERROR) << "Failed to load TensorRT engine!" << endll;
+        return false;
+    }

     ori_size_ = w_ * h_ * 3;
     yuyv_size_ = w_ * h_ * 2;
-    sized_size_ = net_.w * net_.h * 3;
-    rgbf_size_ = w_ * h_ * 3 * sizeof(float);

     cudaError_t err;
     err = cudaMalloc((void **)&dev_ori_, ori_size_);
     // ... 其他分配保持不变 ...
-    err = cudaMalloc((void **)&dev_sized_, sized_size_);
-    check_error(err);
-    err = cudaMalloc((void **)&dev_rgbfp_, rgbf_size_);
-    check_error(err);

     // 相机矩阵相关分配保持不变 ...
 }
```

#### 4.3 修改 run() 推理部分

```diff
     // 预处理到 dev_undis_ 的步骤完全保持不变
     // ...

-    cudaResizePacked(dev_undis_, w_, h_, dev_sized_, net_.w, net_.h);
-    cudaBGR2RGBfp(dev_sized_, dev_rgbfp_, net_.w, net_.h);
-
-    layer l = net_.layers[net_.n - 1];
-    network_predict(net_, dev_rgbfp_, 0);
-    int nboxes = 0;
-    float nms = .45;
-    detection *dets = get_network_boxes(&net_, w_, h_, 0.5, 0.5, 0, 1, &nboxes, 0);
-    if (nms)
-        do_nms_sort(dets, nboxes, l.classes, nms);
-    ball_dets_.clear();
-    post_dets_.clear();
-    for (int i = 0; i < nboxes; i++) { ... }
-    std::sort(ball_dets_.rbegin(), ball_dets_.rend());
-    std::sort(post_dets_.rbegin(), post_dets_.rend());
-    free_detections(dets, nboxes);
+    // 一行替换全部检测逻辑
+    detector_.detect(dev_undis_, w_, h_,
+                     ball_dets_, post_dets_,
+                     ball_id_, post_id_,
+                     ball_prob_, post_prob_,
+                     min_ball_w_, min_ball_h_,
+                     min_post_w_, min_post_h_,
+                     d_w_h_);

     // 以下所有代码（球位置解算、门柱定位、调试图像）完全不变
```

#### 4.4 修改 stop()

```diff
 void Vision::stop()
 {
     if (is_alive_)
     {
         delete_timer();
-        free_network(net_);
+        detector_.release();
         free(camera_src_);
         cudaFree(dev_ori_);
         cudaFree(dev_undis_);
         cudaFree(dev_yuyv_);
         cudaFree(dev_src_);
         cudaFree(dev_bgr_);
-        cudaFree(dev_rgbfp_);
-        cudaFree(dev_sized_);
         // 相机矩阵 cudaFree 保持不变
     }
 }
```

### 步骤 5：修改 CMakeLists.txt

#### vision/CMakeLists.txt

```cmake
add_library(vision
        vision.cpp
        trt_detector.cpp)

target_link_libraries(vision
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        imageproc
        nvinfer          # TensorRT 推理库
        nvinfer_plugin   # TensorRT 插件
        nvonnxparser     # 可选：运行时解析 ONNX
        server
        SL
        pthread
        rt
        stdc++
        cudart)
```

#### src/lib/CMakeLists.txt（可选保留 darknet）

```cmake
# 如果 auto_marker 工具仍需要 darknet，则保留
# add_subdirectory(darknet)
add_subdirectory(imageproc)
add_subdirectory(robot)
add_subdirectory(parser)
```

### 步骤 6：修改配置文件

`src/data/config.conf`：

```diff
-    "net_cfg_file": "data/algorithm/robocup.cfg",
-    "net_weights_file": "data/algorithm/24_V1.weights",
+    "net_engine_file": "data/algorithm/best.engine",
     "net_names_file": "data/algorithm/robocup.names",
```

### 步骤 7：清理 imageproc 的 Darknet 依赖

#### imageproc.hpp

```diff
-#include "darknet/network.h"
+// 不再依赖 darknet
```

#### color/color.cpp

```diff
-#include "darknet/cuda.h"
+#include <cuda_runtime.h>

+// 从 darknet 移出的工具函数
+inline void check_error(cudaError_t status) {
+    if (status != cudaSuccess) {
+        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
+    }
+}
```

或者将 `check_error` 提取到公共头文件 `src/lib/cuda_utils.hpp` 中。

---

## 五、预处理差异详解

这是迁移中**最容易出错**的部分。

### 5.1 YOLOv3 Darknet 预处理

```
原图 BGR (640×480) 
  → cudaResizePacked 直接 resize 到 net.w × net.h (如 416×416)
  → cudaBGR2RGBfp 转为 float RGB [0, 1]
```

**特点**：直接拉伸，不保持宽高比。

### 5.2 YOLOv8 预处理

```
原图 BGR (640×480)
  → Letterbox resize 到 640×640 (保持宽高比，灰色填充)
  → BGR → RGB
  → 归一化到 [0, 1] float32
  → HWC → CHW (NCHW 格式)
```

**关键差异**：

| 项目 | YOLOv3 Darknet | YOLOv8 |
|------|---------------|--------|
| Resize 方式 | 直接拉伸 | **Letterbox (保持宽高比)** |
| 输入尺寸 | 通常 416×416 | 通常 640×640 |
| 颜色通道 | RGB float [0,1] | RGB float [0,1] |
| 数据排列 | CHW (Darknet内部处理) | **CHW (需自行转换)** |
| 填充值 | 无 | **114/255 ≈ 0.447 (灰色)** |

### 5.3 推荐的 Letterbox CUDA Kernel

```cpp
// 需要新增的 CUDA kernel
__global__ void letterbox_preprocess_kernel(
    const unsigned char* src,  // BGR HWC uint8
    float* dst,                // RGB CHW float32
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale, int pad_x, int pad_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int src_x = (int)((x - pad_x) / scale);
    int src_y = (int)((y - pad_y) / scale);

    float r, g, b;
    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h)
    {
        int src_idx = (src_y * src_w + src_x) * 3;
        b = src[src_idx + 0] / 255.0f;
        g = src[src_idx + 1] / 255.0f;
        r = src[src_idx + 2] / 255.0f;
    }
    else
    {
        r = g = b = 114.0f / 255.0f;  // 灰色填充
    }

    // CHW 排列
    int area = dst_w * dst_h;
    dst[0 * area + y * dst_w + x] = r;   // R channel
    dst[1 * area + y * dst_w + x] = g;   // G channel
    dst[2 * area + y * dst_w + x] = b;   // B channel
}
```

---

## 六、后处理差异详解

### 6.1 YOLOv3 输出格式

```
每个检测框: [cx, cy, w, h, objectness, class0, class1, ...]
最终概率 = objectness × class_prob
```

Darknet API 内部完成了解码，`get_network_boxes()` 直接返回像素坐标。

### 6.2 YOLOv8 输出格式

```
输出张量: [1, 4+num_classes, 8400]
每个 anchor: [cx, cy, w, h, class0_prob, class1_prob]
无 objectness，直接取类别概率的最大值
坐标是相对于输入尺寸的像素值（非归一化）
```

### 6.3 坐标还原

由于使用了 letterbox，输出坐标需要还原：

```cpp
// 从网络输出坐标 → 原图像素坐标
float real_cx = (net_cx - pad_x) / scale;
float real_cy = (net_cy - pad_y) / scale;
float real_w  = net_w / scale;
float real_h  = net_h / scale;

// 转为左上角坐标（与原 object_det 格式一致）
int x = (int)(real_cx - real_w / 2.0f);
int y = (int)(real_cy - real_h / 2.0f);
```

### 6.4 NMS 实现

YOLOv8 不自带 NMS，需自行实现（简单 CPU 版即可，8400 个候选框处理很快）：

```cpp
void TRTDetector::nms(std::vector<object_det>& dets, float nms_thresh)
{
    std::sort(dets.rbegin(), dets.rend());  // 按 prob 降序
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); i++)
    {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); j++)
        {
            if (suppressed[j]) continue;
            if (iou(dets[i], dets[j]) > nms_thresh)
                suppressed[j] = true;
        }
    }

    std::vector<object_det> result;
    for (size_t i = 0; i < dets.size(); i++)
        if (!suppressed[i])
            result.push_back(dets[i]);
    dets = std::move(result);
}
```

---

## 七、实施计划与时间线

### 阶段一：准备工作（1-2 天）

- [ ] 1. 整理现有训练数据集，确认 YOLO 格式标注正确
- [ ] 2. 训练 YOLOv8n/YOLOv8s 模型（从预训练权重开始 fine-tune）
- [ ] 3. 导出 ONNX，在 PC 上用 Python 验证推理正确性
- [ ] 4. 在 Orin NX 上转换 TensorRT Engine

### 阶段二：推理封装（2-3 天）

- [ ] 5. 实现 `TRTDetector` 类（加载、推理、后处理）
- [ ] 6. 编写 letterbox 预处理 CUDA kernel
- [ ] 7. 实现 YOLOv8 后处理 + NMS
- [ ] 8. 编写独立测试程序验证推理结果与 Python 一致

### 阶段三：集成替换（1-2 天）

- [ ] 9. 修改 `vision.hpp` / `vision.cpp`（按步骤 3-4）
- [ ] 10. 修改 CMakeLists.txt
- [ ] 11. 清理 imageproc 的 darknet 依赖
- [ ] 12. 修改配置文件

### 阶段四：测试验证（2-3 天）

- [ ] 13. 编译通过，解决链接问题
- [ ] 14. 静态图片测试：对比 YOLOv3 和 YOLOv8 检测结果
- [ ] 15. 实际运行测试：验证球识别距离、门柱定位精度
- [ ] 16. 性能测试：帧率对比（目标 ≥20 FPS）
- [ ] 17. 全流程测试：机器人行走 + 识球 + 踢球

---

## 八、注意事项与常见问题

### 8.1 关键注意事项

1. **类别 ID 必须一致**
   - 原代码 `ball_id_=1`, `post_id_=0`
   - 训练 YOLOv8 时 YAML 中类别顺序必须与此对应
   - 否则会把球当门柱、门柱当球

2. **TensorRT Engine 不可跨平台**
   - `.engine` 文件与 GPU 架构、TensorRT 版本、CUDA 版本绑定
   - 必须在 **目标 Orin NX** 上用 `trtexec` 生成
   - 建议写脚本自动从 `.onnx` 生成 `.engine`（首次启动时）

3. **Letterbox 坐标还原**
   - 这是最容易出 bug 的地方
   - 如果坐标不正确，`odometry()` 测距会全面偏差
   - **验证方法**：在调试图上画检测框，确认框位置准确

4. **输入格式必须是 CHW**
   - Darknet 内部会处理 HWC→CHW
   - TensorRT 需要自行确保输入为 NCHW
   - `cudaBGR2RGBfp()` 输出是 CHW，如果复用需要确认

5. **不要修改 object_det 结构**
   - 所有下游模块依赖此结构
   - 检测器输出必须填充到一样格式的 `ball_dets_` / `post_dets_`

6. **check_error 函数依赖问题**
   - `vision.cpp` 和 `color.cpp` 中多处使用 `check_error()`
   - 此函数定义在 `darknet/cuda.c` 中
   - 如果移除 darknet，需要提取此函数到公共位置

### 8.2 性能预期

| 模型 | Orin NX FP16 推理 | 精度对比 |
|------|------------------|---------|
| YOLOv8n (640) | ~3-5 ms | 略优于 YOLOv3-tiny |
| YOLOv8s (640) | ~6-10 ms | 明显优于 YOLOv3 |
| YOLOv8m (640) | ~15-25 ms | 大幅优于 YOLOv3 |
| 原 YOLOv3 (416) | ~10-15 ms | 基准 |

**推荐**：从 YOLOv8n 或 YOLOv8s 开始，满足实时性要求。

### 8.3 回退方案

建议通过配置文件支持**双检测器切换**：

```json
{
    "detector_type": "tensorrt",
    "net_engine_file": "data/algorithm/best.engine",
    "net_cfg_file": "data/algorithm/robocup.cfg",
    "net_weights_file": "data/algorithm/24_V1.weights"
}
```

在 `vision.cpp` 中根据 `detector_type` 选择 Darknet 或 TensorRT 路径，便于对比测试和紧急回退。

### 8.4 auto_marker 工具处理

`src/tools/auto_marker/main.cpp` 也使用了完整的 Darknet 推理流程。两种处理方式：

- **方案 A**：暂时保留 darknet 库，auto_marker 不改
- **方案 B**：同步迁移，让 auto_marker 也使用 TRTDetector

推荐先用方案 A，主线稳定后再迁移工具。

### 8.5 调试建议

1. **逐步验证**：先用 Python 跑通 ONNX 推理 → 用 C++ TensorRT 跑通 → 再集成到 Vision
2. **保存中间结果**：预处理后的输入张量、网络输出，与 Python 端逐字节对比
3. **检测框可视化**：集成后第一件事是在调试图上验证检测框位置
4. **距离验证**：用已知距离的球验证 `odometry()` 输出是否正确

---

## 九、文件变更汇总

```
新增文件：
  src/controller/player/vision/trt_detector.hpp     ← TensorRT 推理封装
  src/controller/player/vision/trt_detector.cpp     ← 实现
  src/controller/player/vision/letterbox.cu         ← letterbox CUDA kernel (可选)
  src/lib/cuda_utils.hpp                            ← check_error 等公共 CUDA 工具 (可选)
  src/data/algorithm/best.onnx                      ← YOLOv8 ONNX 模型
  src/data/algorithm/best.engine                    ← TensorRT 序列化引擎

修改文件：
  src/controller/player/vision/vision.hpp           ← 用 TRTDetector 替换 network
  src/controller/player/vision/vision.cpp           ← start/run/stop 改动
  src/controller/player/vision/CMakeLists.txt       ← 链接 TensorRT
  src/data/config.conf                              ← 模型路径
  src/lib/imageproc/imageproc.hpp                   ← 移除 darknet include
  src/lib/imageproc/color/color.cpp                 ← check_error 来源替换

可选保留：
  src/lib/darknet/                                  ← 如果 auto_marker 仍需要
```
