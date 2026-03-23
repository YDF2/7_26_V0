# YOLOv3 Darknet → YOLOv8 TensorRT 完整迁移方案

> **平台**：NVIDIA Jetson Orin NX (aarch64, JetPack 5.x/6.x)  
> **推理后端**：TensorRT C++ API（唯一方案）  
> **目标**：替换视觉检测器，保持原代码架构和所有下游模块不变  
> **日期**：2026-02-25

---

## 目录

1. [当前架构完整分析](#一当前架构完整分析)
2. [TensorRT 推理方案设计](#二tensorrt-推理方案设计)
3. [模型训练与转换](#三模型训练与转换)
4. [新增文件：完整实现代码](#四新增文件完整实现代码)
5. [现有文件改动（逐文件逐行）](#五现有文件改动逐文件逐行)
6. [编译构建指南](#六编译构建指南)
7. [测试验证流程](#七测试验证流程)
8. [故障排查手册](#八故障排查手册)
9. [实施计划与时间线](#九实施计划与时间线)

---

## 一、当前架构完整分析

### 1.1 Darknet 检测完整调用链

通过阅读项目源码，Darknet 的调用涉及以下精确位置：

| 阶段 | Darknet API 调用 | 所在函数 | 文件位置 |
|------|-----------------|---------|---------|
| 加载模型 | `parse_network_cfg_custom()` | `Vision::start()` | vision.cpp L489 |
| 加载权重 | `load_weights()` | `Vision::start()` | vision.cpp L490 |
| 设置批次 | `set_batch_network(&net_, 1)` | `Vision::start()` | vision.cpp L491 |
| 优化融合 | `fuse_conv_batchnorm(net_)` | `Vision::start()` | vision.cpp L492 |
| 二值化权重 | `calculate_binary_weights(net_)` | `Vision::start()` | vision.cpp L493 |
| 获取网络尺寸 | `net_.w`, `net_.h` | `Vision::start()` / `Vision::run()` | vision.cpp L497, L188-189 |
| GPU Resize 预处理 | `cudaResizePacked(dev_undis_, w_, h_, dev_sized_, net_.w, net_.h)` | `Vision::run()` | vision.cpp L188 |
| BGR→RGB 浮点 | `cudaBGR2RGBfp(dev_sized_, dev_rgbfp_, net_.w, net_.h)` | `Vision::run()` | vision.cpp L189 |
| 获取最后一层 | `layer l = net_.layers[net_.n - 1]` | `Vision::run()` | vision.cpp L197 |
| GPU 推理 | `network_predict(net_, dev_rgbfp_, 0)` | `Vision::run()` | vision.cpp L198 |
| 获取检测框 | `get_network_boxes(&net_, w_, h_, 0.5, 0.5, 0, 1, &nboxes, 0)` | `Vision::run()` | vision.cpp L201 |
| NMS 排序 | `do_nms_sort(dets, nboxes, l.classes, nms)` | `Vision::run()` | vision.cpp L204 |
| 释放检测结果 | `free_detections(dets, nboxes)` | `Vision::run()` | vision.cpp L244 |
| 释放网络 | `free_network(net_)` | `Vision::stop()` | vision.cpp L546 |

### 1.2 现有预处理流水线（完整链路）

```
相机原始数据 (YUYV/Bayer, camera_w_ × camera_h_)
    │
    ├── [MVCamera] cudaBayer2BGR() → dev_bgr_
    └── [USBCamera] cudaYUYV2BGR() → dev_bgr_
    │
    ▼
cudaResizePacked(dev_bgr_ → dev_ori_)  // 缩放到 w_×h_ (640×480)
    │
    ├── [MVCamera] cudaUndistored(dev_ori_ → dev_undis_)  // 畸变矫正
    └── [USBCamera] cudaMemcpy(dev_ori_ → dev_undis_)     // 直接拷贝
    │
    ▼
dev_undis_ (BGR uint8, 640×480)  ← 这是给可视化和后续模块共用的图像
    │
    ├── cudaResizePacked(dev_undis_ → dev_sized_)  // 缩放到 net_.w × net_.h (416×416)
    └── cudaBGR2RGBfp(dev_sized_ → dev_rgbfp_)    // BGR uint8 → RGB float CHW [0,1]
        │
        ▼
    network_predict(net_, dev_rgbfp_, 0)  // Darknet GPU 推理
```

### 1.3 Darknet 输出解析逻辑（vision.cpp L206-L243）

Darknet 通过 `get_network_boxes()` 返回归一化坐标 `[cx, cy, w, h]`（范围 0~1），然后代码手动转换为像素坐标：

```cpp
// 球检测
int bx = (dets[i].bbox.x - dets[i].bbox.w / 2.0) * w_;  // 左上角 x
int by = (dets[i].bbox.y - dets[i].bbox.h / 2.0) * h_;  // 左上角 y
int bw = dets[i].bbox.w * w_;                             // 宽度像素
int bh = dets[i].bbox.h * h_ + 1;                         // 高度像素

// 门柱检测
int px = (dets[i].bbox.x - dets[i].bbox.w / 2.0) * w_;
int py = (dets[i].bbox.y - dets[i].bbox.h / 2.0) * h_;
int pw = dets[i].bbox.w * w_;
int ph = dets[i].bbox.h * h_;
```

**关键**：Darknet 的 `get_network_boxes()` 内部已经将坐标从网络输入尺寸映射回了原图尺寸（通过传入的 `w_`, `h_` 参数），所以输出是相对于 640×480 的归一化值。

### 1.4 `object_det` 结构体（model.hpp L110-L119）

```cpp
struct object_det {
    int id;              // 类别ID (0=门柱post, 1=足球ball)
    float prob;          // 置信度
    int x, y, w, h;     // 像素坐标 (左上角 x,y + 宽高 w,h)
    object_det(int i=0, float p=1, int x_=0, int y_=0, int w_=0, int h_=0);
    bool operator< (const object_det &obj) { return prob < obj.prob; }
};
```

**此结构不可修改**，所有下游模块均依赖它。

### 1.5 检测参数（从 config.conf 读取）

| 参数 | 配置路径 | 当前值 | 含义 |
|------|---------|-------|------|
| `ball_id_` | 硬编码 | 1 | 足球类别 ID |
| `post_id_` | 硬编码 | 0 | 门柱类别 ID |
| `ball_prob_` | `detection.ball` | 0.6 | 足球置信度阈值 |
| `post_prob_` | `detection.post` | 0.3 | 门柱置信度阈值 |
| `min_ball_w_` | `detection.ball_w` | 5 | 足球最小宽度 |
| `min_ball_h_` | `detection.ball_h` | 5 | 足球最小高度 |
| `min_post_w_` | `detection.post_w` | 10 | 门柱最小宽度 |
| `min_post_h_` | `detection.post_h` | 15 | 门柱最小高度 |
| `d_w_h_` | 硬编码 | 0.3 | 球宽高比容差 |

### 1.6 `check_error()` 函数依赖分析

`check_error()` 定义在 `src/lib/darknet/cuda.c` L27-L43，声明在 `src/lib/darknet/cuda.h` L39。以下文件使用了它：

| 文件 | 调用次数 | 来源 |
|------|---------|------|
| `vision.cpp` | 20+ 次 | 隐式通过链接 darknet 库 |
| `color/color.cpp` | 3 次 | `#include "darknet/cuda.h"` |
| `auto_marker/main.cpp` | 4 次 | `#include "darknet/cuda.h"` (隐式) |

### 1.7 完整文件影响范围

| 文件 | 改动程度 | Darknet 依赖点 |
|------|---------|---------------|
| `src/controller/player/vision/vision.hpp` | **大** | `#include "darknet/network.h"`, 成员 `network net_` |
| `src/controller/player/vision/vision.cpp` | **大** | `#include "darknet/parser.h"`, `start()/run()/stop()` 全部推理逻辑 |
| `src/controller/player/vision/CMakeLists.txt` | **中** | 链接 `darknet` 库 |
| `src/lib/imageproc/imageproc.hpp` | **小** | `#include "darknet/network.h"`（未使用任何 darknet 类型，纯多余包含） |
| `src/lib/imageproc/color/color.cpp` | **小** | `#include "darknet/cuda.h"` 仅用 `check_error()` |
| `src/data/config.conf` | **小** | `net_cfg_file`, `net_weights_file` 配置项 |
| `src/lib/CMakeLists.txt` | **小** | `add_subdirectory(darknet)` 保留给 auto_marker |
| `src/tools/auto_marker/main.cpp` | **暂不改** | 完整 Darknet 依赖，保留原样 |

### 1.8 绝对不受影响的模块

以下模块**只依赖** `ball_dets_` / `post_dets_`（`std::vector<object_det>`），与检测器实现完全解耦：

- `odometry()` 测距逻辑（vision.cpp 内部函数）
- `camera2self()` 坐标变换
- 世界模型 `WorldModel` (`WM->set_ball_pos()`)
- 自定位 `SelfLocalization` (`SL->update()`)
- 调试图像绘制与发送 (`send_image()`)
- 所有通信协议、策略、运动控制

---

## 二、TensorRT 推理方案设计

### 2.1 为什么选 TensorRT C++ API

| 优势 | 说明 |
|------|------|
| **零额外依赖** | JetPack 自带 TensorRT，无需安装 ONNX Runtime |
| **最低延迟** | 直接操作 GPU 内存，FP16/INT8 硬件加速 |
| **与现有管线无缝衔接** | 输入仍是 GPU 上的 `unsigned char*`，不需要 CPU↔GPU 拷贝 |
| **内存效率** | Engine 直接从文件加载，不需解析 ONNX 图 |

### 2.2 架构设计原则

**核心思想：用一个 `TRTDetector` 类完整替换 Darknet，对外接口与原有逻辑完全兼容。**

```
【替换前】
dev_undis_ (BGR uint8, GPU)
    → cudaResizePacked() → dev_sized_
    → cudaBGR2RGBfp()   → dev_rgbfp_
    → network_predict()
    → get_network_boxes() + do_nms_sort()
    → 手动解析到 ball_dets_ / post_dets_
    → free_detections()

【替换后】
dev_undis_ (BGR uint8, GPU)
    → detector_.detect(dev_undis_, w_, h_, ball_dets_, post_dets_, ...)
       内部完成：letterbox预处理 → TensorRT推理 → YOLOv8后处理+NMS → 填充结果
```

### 2.3 `TRTDetector` 类职责

| 方法 | 职责 | 对应原 Darknet 操作 |
|------|------|-------------------|
| `load()` | 加载 .engine 文件，分配 GPU 缓冲 | `parse_network_cfg_custom` + `load_weights` + `set_batch_network` + `fuse_conv_batchnorm` + `calculate_binary_weights` + `cudaMalloc(dev_sized_)` + `cudaMalloc(dev_rgbfp_)` |
| `detect()` | 预处理+推理+后处理，输出填充到 ball_dets/post_dets | `cudaResizePacked` + `cudaBGR2RGBfp` + `network_predict` + `get_network_boxes` + `do_nms_sort` + 解析循环 + `free_detections` |
| `release()` | 释放所有资源 | `free_network` + `cudaFree(dev_sized_)` + `cudaFree(dev_rgbfp_)` |

### 2.4 预处理差异对比

| 项目 | YOLOv3 Darknet（当前） | YOLOv8 TensorRT（替换后） |
|------|----------------------|--------------------------|
| Resize 方式 | **直接拉伸**到 net_w × net_h | **Letterbox**（保持宽高比，灰色填充） |
| 输入尺寸 | 通常 416×416 | 通常 640×640 |
| 颜色空间 | RGB float [0,1] CHW | RGB float [0,1] CHW |
| 填充值 | 无 | 114/255 ≈ 0.447（灰色） |
| 数据排列 | CHW（`cudaBGR2RGBfp` 已实现） | CHW（需自行在 letterbox kernel 中完成） |

### 2.5 后处理差异对比

| 项目 | YOLOv3 Darknet | YOLOv8 TensorRT |
|------|---------------|-----------------|
| 输出格式 | `detection` 结构体数组（API 内部解码） | 原始张量 `[1, 4+nc, 8400]` |
| Objectness | 有（`prob = objectness × class_prob`） | **无**（直接输出 class_prob） |
| 坐标格式 | 归一化 cx,cy,w,h（相对原图） | **像素** cx,cy,w,h（相对网络输入 640×640） |
| NMS | `do_nms_sort()`（API 内置） | **需自行实现** |
| 坐标映射 | API 内部完成 | **需通过 letterbox 反算回原图** |

---

## 三、模型训练与转换

### 3.1 训练 YOLOv8 模型

```bash
# 安装 ultralytics
pip install ultralytics

# 使用现有标注数据训练
yolo detect train data=robocup.yaml model=yolov8n.pt epochs=200 imgsz=640
```

`robocup.yaml`：
```yaml
names:
  0: post    # 必须对应原 post_id_ = 0
  1: ball    # 必须对应原 ball_id_ = 1
nc: 2
train: ./dataset/train/images
val: ./dataset/val/images
```

> **⚠️ 类别 ID 一致性是最关键的检查项**：原代码 `ball_id_=1`, `post_id_=0`，训练时必须严格对应。

### 3.2 导出 ONNX

```bash
yolo export model=best.pt format=onnx imgsz=640 opset=11 simplify=True
```

### 3.3 验证 ONNX 输入输出

```bash
python3 -c "
import onnx
model = onnx.load('best.onnx')
for inp in model.graph.input:
    print('Input:', inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim])
for out in model.graph.output:
    print('Output:', out.name, [d.dim_value for d in out.type.tensor_type.shape.dim])
"
```

预期输出（2 类模型）：
```
Input: images [1, 3, 640, 640]       # NCHW, float32, [0,1]
Output: output0 [1, 6, 8400]         # 6 = 4坐标 + 2类别概率
```

### 3.4 在 Orin NX 上转换 TensorRT Engine

**⚠️ 必须在目标设备上执行，Engine 与 GPU 架构/TensorRT 版本绑定，不可跨平台移植。**

```bash
# FP16 精度（推荐：速度与精度最佳平衡）
/usr/src/tensorrt/bin/trtexec \
    --onnx=best.onnx \
    --saveEngine=best.engine \
    --fp16 \
    --workspace=1024 \
    --verbose

# 验证 Engine 是否正常
/usr/src/tensorrt/bin/trtexec --loadEngine=best.engine --verbose
```

生成的 `best.engine` 放到 `src/data/algorithm/best.engine`。

### 3.5 Python 端验证推理正确性（可选但强烈推荐）

在集成到 C++ 之前，先用 Python 验证推理结果是否正确：

```python
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def letterbox(img, new_shape=(640, 640)):
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    pad_h = (new_shape[0] - new_h) // 2
    pad_w = (new_shape[1] - new_w) // 2
    img_padded = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)
    img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized
    return img_padded, scale, pad_w, pad_h

# 加载测试图片
img = cv2.imread("test.jpg")
img_pre, scale, pad_x, pad_y = letterbox(img)
blob = img_pre[:,:,::-1].transpose(2,0,1).astype(np.float32) / 255.0  # BGR→RGB, HWC→CHW
blob = np.expand_dims(blob, 0)

# ... TensorRT 推理 ...
# 输出 shape: [1, 6, 8400]
# 转置为 [8400, 6]
output = output.reshape(6, 8400).T

for det in output:
    cx, cy, w, h, prob_post, prob_ball = det
    max_prob = max(prob_post, prob_ball)
    if max_prob < 0.5:
        continue
    # 坐标还原
    real_cx = (cx - pad_x) / scale
    real_cy = (cy - pad_y) / scale
    real_w = w / scale
    real_h = h / scale
    x1 = int(real_cx - real_w / 2)
    y1 = int(real_cy - real_h / 2)
    print(f"Class={'ball' if prob_ball>prob_post else 'post'}, "
          f"prob={max_prob:.3f}, box=({x1},{y1},{int(real_w)},{int(real_h)})")
```

---

## 四、新增文件：完整实现代码

### 4.1 新建 `src/lib/cuda_utils.hpp` — 公共 CUDA 工具函数

将 `check_error()` 从 darknet 中提取出来，避免非 darknet 模块对 darknet 的依赖：

```cpp
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
```

---

### 4.2 新建 `src/controller/player/vision/trt_detector.hpp`

```cpp
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
    // ---- TensorRT 核心对象 ----
    nvinfer1::IRuntime*          runtime_ = nullptr;
    nvinfer1::ICudaEngine*       engine_  = nullptr;
    nvinfer1::IExecutionContext*  context_ = nullptr;

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
```

---

### 4.3 新建 `src/controller/player/vision/trt_detector.cpp`

```cpp
#include "trt_detector.hpp"
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
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
extern void cudaLetterboxPreprocess(
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

    // 3. 获取输入维度 [1, 3, H, W]
    //    TensorRT 8.x 使用 name-based binding
    //    兼容写法：先试 getBindingDimensions(0)
    auto input_dims = engine_->getBindingDimensions(0);
    if (input_dims.nbDims != 4)
    {
        LOG(LOG_ERROR) << "TRTDetector: Unexpected input dims: " << input_dims.nbDims << endll;
        return false;
    }
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];

    // 4. 获取输出维度 [1, 4+nc, num_anchors]
    auto output_dims = engine_->getBindingDimensions(1);
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

    // 5. 分配 GPU 缓冲区
    cudaMalloc(&dev_input_,  3 * input_h_ * input_w_ * sizeof(float));
    cudaMalloc(&dev_output_, output_size_ * sizeof(float));

    // 6. 分配 CPU 输出缓冲
    host_output_ = new float[output_size_];

    // 7. 创建 CUDA stream
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
    void* bindings[2] = { dev_input_, dev_output_ };
    context_->enqueueV2(bindings, stream_, nullptr);

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
    if (context_) { context_->destroy(); context_ = nullptr; }
    if (engine_)  { engine_->destroy();  engine_  = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }

    if (dev_input_)   { cudaFree(dev_input_);   dev_input_  = nullptr; }
    if (dev_output_)  { cudaFree(dev_output_);  dev_output_ = nullptr; }
    if (host_output_) { delete[] host_output_;   host_output_ = nullptr; }

    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}
```

---

### 4.4 新建 `src/controller/player/vision/trt_preprocess.cu` — Letterbox CUDA Kernel

```cuda
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
```

---

## 五、现有文件改动（逐文件逐行）

### 5.1 文件 `src/controller/player/vision/vision.hpp`

**改动 1**：替换 darknet 头文件为 TRTDetector 头文件

```diff
 #include "common.hpp"
 #include "singleton.hpp"
-#include "darknet/network.h"
+#include "trt_detector.hpp"
 #include "tcp.hpp"
```

**改动 2**：替换 `network net_` 成员为 `TRTDetector detector_`

```diff
     bool is_busy_;
     image_send_type img_sd_type_;
 
-    network net_;
+    TRTDetector detector_;
```

**改动 3**：移除 `dev_sized_`、`dev_rgbfp_`、`sized_size_`、`rgbf_size_` 成员（全部由 TRTDetector 内部管理）

```diff
     unsigned char *dev_ori_;
-    unsigned char *dev_sized_;
     unsigned char *dev_undis_;
     unsigned char *dev_yuyv_;
     unsigned char *camera_src_;
-    float *dev_rgbfp_;
     int src_size_;
     int bgr_size_;
     int ori_size_;
     int yuyv_size_;
-    int sized_size_;
-    int rgbf_size_;
```

完整改动后的 vision.hpp：

```cpp
#ifndef __VISION_HPP
#define __VISION_HPP

#include <queue>
#include <atomic>
#include <mutex>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "timer.hpp"
#include "observer.hpp"
#include "sensor/camera.hpp"
#include "configuration.hpp"
#include "options/options.hpp"
#include "common.hpp"
#include "singleton.hpp"
#include "trt_detector.hpp"       // ← 替换 "darknet/network.h"
#include "tcp.hpp"
#include "common.hpp"
#include "math/math.hpp"
#include "sensor/imu.hpp"
#include "sensor/motor.hpp"
#include "localization/localization.h"

class Vision: public Timer, public Subscriber, public Singleton<Vision>
{
public:
    Vision();
    ~Vision();
    int w(){return w_;}
    int h(){return h_;}
    void updata(const pub_ptr &pub, const int &type);
    bool start();
    void stop();
    void set_camera_info(const camera_info &para);
    void get_point_dis(int x, int y);

    void set_img_send_type(image_send_type t)
    {
        img_sd_type_ = t;
    }

public:
    std::atomic_bool localization_;
    std::atomic_bool can_see_post_;
private:
    Eigen::Vector2d odometry(const Eigen::Vector2i &pos, const robot_math::TransformMatrix &camera_matrix);
    Eigen::Vector2d camera2self(const Eigen::Vector2d &pos, double head_yaw);
    Imu::imu_data imu_data_vison;
    float dir=0;
private:
    void run();
    void send_image(const cv::Mat &src);
    
    std::queue<Imu::imu_data> imu_datas_;
    std::queue< std::vector<double> > foot_degs_, head_degs_;
    std::queue<int> spfs_;
    
    float head_yaw_, head_pitch_;

    Eigen::Vector2d odometry_offset_;

    bool detect_filed_;
    bool use_mv_;
    int p_count_;
    int w_, h_;
    int camera_w_,  camera_h_, camera_size_;
    std::map<std::string, camera_info> camera_infos_;
    camera_param params_;
    robot_math::TransformMatrix camera_matrix_;
    
    std::vector<object_det> ball_dets_, post_dets_; 
    float d_w_h_;
    int ball_id_, post_id_;
    float ball_prob_, post_prob_;
    int min_ball_w_, min_ball_h_;
    int min_post_w_, min_post_h_;

    int cant_see_ball_count_;
    int can_see_post_count_;

    bool is_busy_;
    image_send_type img_sd_type_;

    TRTDetector detector_;          // ← 替换 network net_

    unsigned char *dev_src_;
    unsigned char *dev_bgr_;
    unsigned char *dev_ori_;
    // dev_sized_ 已移除（由 TRTDetector 内部管理）
    unsigned char *dev_undis_;
    unsigned char *dev_yuyv_;
    unsigned char *camera_src_;
    // dev_rgbfp_ 已移除（由 TRTDetector 内部管理）
    int src_size_;
    int bgr_size_;
    int ori_size_;
    int yuyv_size_;
    // sized_size_, rgbf_size_ 已移除

    cv::Mat camK;
    cv::Mat newCamK;
    cv::Mat invCamK;
    cv::Mat D;
    cv::Mat R;

    cv::Mat mapx;
    cv::Mat mapy;

    float *pCamKData;
    float *pInvNewCamKData;
    float *pDistortData;
    float *pMapxData;
    float *pMapyData;
    
    mutable std::mutex frame_mtx_, imu_mtx_;
};

#define VISION Vision::instance()

#endif
```

---

### 5.2 文件 `src/controller/player/vision/vision.cpp`

#### 改动 1: 头文件（第 1-4 行）

```diff
 #include "vision.hpp"
 #include "parser/parser.hpp"
-#include "darknet/parser.h"
 #include <cuda_runtime.h>
```

移除 `"darknet/parser.h"`，不需要替换为其他文件（TRTDetector 的头文件已通过 vision.hpp 引入）。

#### 改动 2: `start()` 函数 — 替换模型加载（第 489-513 行）

原代码：
```cpp
bool Vision::start()
{
    net_.gpu_index = 0;
    net_ = parse_network_cfg_custom((char *)CONF->get_config_value<string>("net_cfg_file").c_str(), 1);
    load_weights(&net_, (char *)CONF->get_config_value<string>("net_weights_file").c_str());
    set_batch_network(&net_, 1);
    fuse_conv_batchnorm(net_);
    calculate_binary_weights(net_);
    srand(2222222);

    ori_size_ = w_ * h_ * 3;
    yuyv_size_ = w_ * h_ * 2;
    sized_size_ = net_.w * net_.h * 3;
    rgbf_size_ = w_ * h_ * 3 * sizeof(float);

    cudaError_t err;
    err = cudaMalloc((void **)&dev_ori_, ori_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_undis_, ori_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_yuyv_, yuyv_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_sized_, sized_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_rgbfp_, rgbf_size_);
    check_error(err);
```

替换为：
```cpp
bool Vision::start()
{
    // ---- 加载 TensorRT 引擎（替换原 Darknet 加载）----
    std::string engine_path = CONF->get_config_value<string>("net_engine_file");
    if (!detector_.load(engine_path, 2))  // 2 个类别: post=0, ball=1
    {
        LOG(LOG_ERROR) << "Failed to load TensorRT engine: " << engine_path << endll;
        return false;
    }

    ori_size_ = w_ * h_ * 3;
    yuyv_size_ = w_ * h_ * 2;
    // sized_size_ 和 rgbf_size_ 不再需要（TRTDetector 内部管理）

    cudaError_t err;
    err = cudaMalloc((void **)&dev_ori_, ori_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_undis_, ori_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_yuyv_, yuyv_size_);
    check_error(err);
    // dev_sized_ 和 dev_rgbfp_ 的分配已移除
```

其余 `start()` 中的相机矩阵 cudaMalloc / cudaMemcpy 代码**完全保持不变**。

#### 改动 3: `run()` 函数 — 替换推理部分（第 188-244 行）

原代码（从 `dev_undis_` 准备完毕后开始）：
```cpp
        // cudaBGR2YUV422(dev_undis_, dev_yuyv_, w_, h_);
        cudaResizePacked(dev_undis_, w_, h_, dev_sized_, net_.w, net_.h);
        cudaBGR2RGBfp(dev_sized_, dev_rgbfp_, net_.w, net_.h); // 转为浮点型供神经网络使用
        /*
        const int *fieldBorders;
        if(detect_filed_)
        {
            detector_->process(dev_yuyv_);
            fieldBorders = detector_->getBorder();
        }
        */
        layer l = net_.layers[net_.n - 1];
        network_predict(net_, dev_rgbfp_, 0);
        int nboxes = 0;
        float nms = .45;
        detection *dets = get_network_boxes(&net_, w_, h_, 0.5, 0.5, 0, 1, &nboxes, 0);

        if (nms)
            do_nms_sort(dets, nboxes, l.classes, nms);
        ball_dets_.clear();
        post_dets_.clear();
        for (int i = 0; i < nboxes; i++)
        {
            if (dets[i].prob[ball_id_] > dets[i].prob[post_id_])
            {
                if (dets[i].prob[ball_id_] >= ball_prob_)
                {
                    int bx = (dets[i].bbox.x - dets[i].bbox.w / 2.0) * w_;
                    int by = (dets[i].bbox.y - dets[i].bbox.h / 2.0) * h_;
                    int bw = dets[i].bbox.w * w_, bh = dets[i].bbox.h * h_ + 1;
                    float w_h = (float)bw / (float)bh;
                    if (bw >= min_ball_w_ && bh >= min_ball_h_ && fabs(w_h - 1.0) < d_w_h_)
                    {
                        ball_dets_.push_back(object_det(ball_id_, dets[i].prob[ball_id_], bx, by, bw, bh));
                    }
                }
            }
            else
            {
                if (dets[i].prob[post_id_] >= post_prob_)
                {
                    int px = (dets[i].bbox.x - dets[i].bbox.w / 2.0) * w_;
                    int py = (dets[i].bbox.y - dets[i].bbox.h / 2.0) * h_;
                    int pw = dets[i].bbox.w * w_, ph = dets[i].bbox.h * h_;
                    if (pw >= min_post_w_ && ph >= min_post_h_)
                    {
                        post_dets_.push_back(object_det(post_id_, dets[i].prob[post_id_], px, py, pw, ph));
                    }
                }
            }
        }
        std::sort(ball_dets_.rbegin(), ball_dets_.rend());
        std::sort(post_dets_.rbegin(), post_dets_.rend());
        free_detections(dets, nboxes);
```

替换为（仅 3 行）：
```cpp
        // cudaBGR2YUV422(dev_undis_, dev_yuyv_, w_, h_);

        // ---- TensorRT 检测（一行替换全部 Darknet 推理逻辑）----
        detector_.detect(dev_undis_, w_, h_,
                         ball_dets_, post_dets_,
                         ball_id_, post_id_,
                         ball_prob_, post_prob_,
                         min_ball_w_, min_ball_h_,
                         min_post_w_, min_post_h_,
                         d_w_h_);
```

**`run()` 函数中从 `if (OPTS->use_robot())` 开始的所有后续代码完全不变**，包括：
- 球位置解算 (`odometry`, `camera2self`)
- 世界模型更新 (`WM->set_ball_pos()`)
- 门柱定位 (`SL->update()`)
- 调试图像 (`OPTS->use_debug()` 段)

#### 改动 4: `stop()` 函数 — 替换资源释放（第 543-558 行）

原代码：
```cpp
void Vision::stop()
{
    if (is_alive_)
    {
        delete_timer();
        free_network(net_);
        free(camera_src_);
        cudaFree(dev_ori_);
        cudaFree(dev_undis_);
        cudaFree(dev_yuyv_);
        cudaFree(dev_src_);
        cudaFree(dev_bgr_);
        cudaFree(dev_rgbfp_);
        cudaFree(dev_sized_);

        cudaFree(pCamKData);
        ...
```

替换为：
```cpp
void Vision::stop()
{
    if (is_alive_)
    {
        delete_timer();
        detector_.release();      // ← 替换 free_network(net_)
        free(camera_src_);
        cudaFree(dev_ori_);
        cudaFree(dev_undis_);
        cudaFree(dev_yuyv_);
        cudaFree(dev_src_);
        cudaFree(dev_bgr_);
        // cudaFree(dev_rgbfp_) 已移除
        // cudaFree(dev_sized_) 已移除

        cudaFree(pCamKData);
        ...
```

#### 改动 5: `check_error` 来源

移除 `darknet/parser.h` 后，`vision.cpp` 中的 `check_error()` 不再来自 darknet。有两种解决方式：

**方案 A（推荐）**：在 vision.cpp 顶部添加 `#include "cuda_utils.hpp"` 并定义兼容函数：

```cpp
#include "vision.hpp"
#include "parser/parser.hpp"
#include <cuda_runtime.h>
#include "server/server.hpp"
// ...

// check_error 兼容函数（替代 darknet/cuda.h 中的版本）
static inline void check_error(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
        assert(0);
    }
}
```

**方案 B**：使用 `cuda_utils.hpp` 中的 `check_error_cuda()` 并全局替换调用名。

**推荐方案 A**，改动最小，在 vision.cpp 文件内定义一个 static inline 同名函数即可。

---

### 5.3 文件 `src/controller/player/vision/CMakeLists.txt`

原代码：
```cmake
add_library(vision
        vision.cpp)

target_link_libraries(vision
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        imageproc
        darknet
        server
        SL
        pthread
        rt
        stdc++
        cudart)
```

替换为：
```cmake
add_library(vision
        vision.cpp
        trt_detector.cpp
        trt_preprocess.cu)

target_link_libraries(vision
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        imageproc
        nvinfer            # TensorRT 推理核心库
        nvinfer_plugin     # TensorRT 插件（某些层需要）
        server
        SL
        pthread
        rt
        stdc++
        cudart)
```

**变更说明**：
- 添加了 `trt_detector.cpp` 和 `trt_preprocess.cu`
- `darknet` → `nvinfer` + `nvinfer_plugin`
- 不需要 `nvonnxparser`（我们直接加载 .engine，不在运行时解析 ONNX）

---

### 5.4 文件 `src/lib/imageproc/imageproc.hpp`

原代码：
```cpp
#include <memory>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "darknet/network.h"
#include "model.hpp"
```

替换为：
```cpp
#include <memory>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
// "darknet/network.h" 已移除（从未使用 darknet 类型）
#include "model.hpp"
```

**说明**：`imageproc.hpp` 中 `#include "darknet/network.h"` 是多余的，因为该文件没有使用任何 darknet 类型。所有函数都是纯 CUDA + OpenCV 接口。

---

### 5.5 文件 `src/lib/imageproc/color/color.cpp`

原代码第 1-5 行：
```cpp
#include "color.hpp"
#include <fstream>
#include <cuda_runtime.h>
#include "darknet/cuda.h"
```

替换为：
```cpp
#include "color.hpp"
#include <fstream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

// check_error 本地定义（替代 darknet/cuda.h 中的版本）
static void check_error(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        fprintf(stderr, "CUDA Error: %s\n", s);
        assert(0);
    }
}
```

**说明**：`color.cpp` 仅使用了 darknet 的 `check_error()` 函数（3 处调用），定义一个本地同名函数即可完全解除依赖。`color.cu` 不使用任何 darknet 符号，无需修改。

---

### 5.6 文件 `src/data/config.conf`

原代码：
```json
    "net_cfg_file": "data/algorithm/robocup.cfg",
    "net_weights_file": "data/algorithm/24_V1.weights",
    "net_names_file": "data/algorithm/robocup.names",
```

替换为：
```json
    "net_engine_file": "data/algorithm/best.engine",
    "net_names_file": "data/algorithm/robocup.names",
```

**说明**：
- 移除 `net_cfg_file` 和 `net_weights_file`（Darknet 专用）
- 新增 `net_engine_file` 指向 TensorRT 序列化引擎
- 保留 `net_names_file`（可选，用于日志/调试时显示类别名称）

> **注意**：`bin/aarch64/data/config.conf`（部署目录）也需要同步修改。

---

### 5.7 文件 `src/lib/CMakeLists.txt`

原代码：
```cmake
add_subdirectory(darknet)
add_subdirectory(imageproc)
add_subdirectory(robot)
add_subdirectory(parser)
```

**保持不变**。因为 `auto_marker` 工具仍然需要 darknet 库。darknet 只会在 x86_64 构建时被 auto_marker 使用（参见 `src/CMakeLists.txt` 的条件编译），不影响 aarch64 目标。

如果将来 auto_marker 也完成迁移，可以移除 `add_subdirectory(darknet)`。

---

### 5.8 文件 `src/tools/auto_marker/main.cpp` — 暂不修改

`auto_marker` 是独立的标注辅助工具，只在 x86_64 PC 上构建运行（`CMakeLists.txt` 有条件判断）。保留其完整的 Darknet 依赖，待主线稳定后再迁移。

---

## 六、编译构建指南

### 6.1 前置依赖（Orin NX 上）

JetPack 5.x / 6.x 自带以下组件，无需额外安装：
- CUDA Toolkit (≥ 11.4)
- TensorRT (≥ 8.5)
- cuDNN
- OpenCV (≥ 4.x)

验证 TensorRT 可用：
```bash
dpkg -l | grep tensorrt
ls /usr/lib/aarch64-linux-gnu/libnvinfer.so*
```

### 6.2 头文件路径

如果 CMake 找不到 TensorRT 头文件，在顶层 `CMakeLists.txt` 中添加：

```cmake
# 在 include_directories("/usr/local/cuda/targets/aarch64-linux/include") 之后
include_directories("/usr/include/aarch64-linux-gnu")
```

通常 JetPack 已将 NvInfer.h 安装到 `/usr/include/aarch64-linux-gnu/`。

### 6.3 链接库路径

TensorRT 库通常在 `/usr/lib/aarch64-linux-gnu/`，CMake 默认搜索路径应已包含。如果链接失败，添加：

```cmake
link_directories("/usr/lib/aarch64-linux-gnu")
```

### 6.4 编译命令

```bash
# aarch64 交叉编译（原有方式不变）
python3 aarch64-build.py

# 或手动
mkdir -p aarch64-build && cd aarch64-build
cmake .. -DCROSS=ON
make -j$(nproc)
make install
```

### 6.5 常见编译错误及解决

| 错误 | 原因 | 解决 |
|------|------|------|
| `NvInfer.h: No such file` | TensorRT 头文件路径未包含 | 添加 `include_directories` |
| `undefined reference to nvinfer1::*` | 链接库缺失 | 确认 `nvinfer` 在 `target_link_libraries` 中 |
| `undefined reference to check_error` | 移除 darknet 后缺函数 | 在 vision.cpp / color.cpp 中添加本地定义 |
| `multiple definition of check_error` | 同时链接 darknet 和本地定义 | 本地定义使用 `static` 限定 |
| `.cu file not compiled` | CMake 未启用 CUDA 语言 | 确认顶层 `project()` 包含 `CUDA` |

---

## 七、测试验证流程

### 7.1 阶段一：独立推理验证

**目标**：验证 TRTDetector 在脱离 Vision 模块时能正确推理。

编写独立测试程序 `test_trt_detector.cpp`：

```cpp
#include "trt_detector.hpp"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cstdio>

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <engine_path> <image_path>\n", argv[0]);
        return 1;
    }

    TRTDetector detector;
    if (!detector.load(argv[1], 2)) {
        printf("Failed to load engine\n");
        return 1;
    }

    cv::Mat img = cv::imread(argv[2]);
    int w = img.cols, h = img.rows;

    // 上传到 GPU
    unsigned char* dev_bgr;
    cudaMalloc(&dev_bgr, w * h * 3);
    cudaMemcpy(dev_bgr, img.data, w * h * 3, cudaMemcpyHostToDevice);

    std::vector<object_det> ball_dets, post_dets;
    detector.detect(dev_bgr, w, h,
                    ball_dets, post_dets,
                    1, 0,        // ball_id=1, post_id=0
                    0.6f, 0.3f,  // thresholds
                    5, 5, 10, 15,
                    0.3f);

    printf("Balls: %zu, Posts: %zu\n", ball_dets.size(), post_dets.size());
    for (auto& d : ball_dets)
        printf("  Ball: prob=%.3f x=%d y=%d w=%d h=%d\n", d.prob, d.x, d.y, d.w, d.h);
    for (auto& d : post_dets)
        printf("  Post: prob=%.3f x=%d y=%d w=%d h=%d\n", d.prob, d.x, d.y, d.w, d.h);

    // 在图上画框
    for (auto& d : ball_dets)
        cv::rectangle(img, cv::Point(d.x, d.y), cv::Point(d.x+d.w, d.y+d.h), cv::Scalar(255,0,0), 2);
    for (auto& d : post_dets)
        cv::rectangle(img, cv::Point(d.x, d.y), cv::Point(d.x+d.w, d.y+d.h), cv::Scalar(0,0,255), 2);
    cv::imwrite("result.jpg", img);

    cudaFree(dev_bgr);
    detector.release();
    return 0;
}
```

### 7.2 阶段二：与 Python 结果逐字节对比

1. 用 Python 推理同一张图片，保存预处理张量和输出张量
2. 在 C++ 中保存 `dev_input_`（cudaMemcpy 到文件）和 `host_output_`
3. 逐元素对比，偏差应 < 1e-3

### 7.3 阶段三：集成验证

1. 编译通过后，先在静态图片模式下运行
2. 在调试回传图像上确认检测框位置正确
3. 对比 YOLOv3 和 YOLOv8 对同一场景的检测结果

### 7.4 阶段四：实际运行验证

| 测试项 | 验证方法 | 通过标准 |
|--------|---------|---------|
| 帧率 | 运行时输出推理耗时 | ≥ 20 FPS |
| 球检测距离 | 已知距离放球，读 odometry 输出 | 误差 < 10% |
| 球宽高比 | 查看 ball_dets 的 w/h | |w/h-1| < 0.3 |
| 门柱定位 | 站在已知位置，查看 SL 输出 | 位置误差 < 0.3m |
| 漏检率 | 5m 内放球，统计识别率 | > 95% |
| 误检率 | 无球场景运行 | 误检帧 < 1% |

---

## 八、故障排查手册

### 8.1 坐标偏移（检测框位置不准）

**最常见原因**：Letterbox 坐标还原错误。

排查步骤：
1. 打印 `scale_`, `pad_x_`, `pad_y_` 值
2. 用一张已知物体位置的图片测试
3. 对比 Python 推理的坐标和 C++ 的坐标
4. 检查 `(cx - pad_x) / scale` 公式是否正确

### 8.2 类别反了（球当门柱）

**原因**：训练时类别 ID 与代码不一致。

排查步骤：
1. 检查 `robocup.yaml` 中 `0: post, 1: ball` 是否正确
2. 在 postprocess 中打印 `prob_post` 和 `prob_ball` 值
3. 确认 `host_output_[(4+0)*num_anchors_+i]` 对应 post

### 8.3 推理输出全零

排查步骤：
1. 确认 .engine 文件是在目标设备上生成的
2. 检查 TensorRT 版本是否匹配
3. 打印 `dev_input_` 的前几个值，确认预处理输出非零
4. 测试：`trtexec --loadEngine=best.engine --verbose`

### 8.4 Engine 加载失败

排查步骤：
1. `.engine` 文件是否损坏（文件大小合理？通常 5-50 MB）
2. TensorRT 版本是否一致（生成和运行必须同版本）
3. GPU 架构是否匹配（Orin NX = SM 8.7）

### 8.5 性能不达标

排查步骤：
1. 确认使用了 FP16（`--fp16` 参数）
2. 检查是否有其他 GPU 任务竞争资源
3. 使用 `cudaEvent` 精确测量推理耗时（不包含预处理和后处理）
4. 考虑使用 YOLOv8n（最轻量）而非 YOLOv8s

---

## 九、实施计划与时间线

### 阶段一：准备工作（1-2 天）

- [ ] 整理现有训练数据集（YOLO 格式标注，确认 class 0=post, 1=ball）
- [ ] 训练 YOLOv8n 模型（`yolo detect train`）
- [ ] 导出 ONNX（`yolo export format=onnx`）
- [ ] 在 PC 上用 Python 验证推理正确性
- [ ] 在 Orin NX 上转换 TensorRT Engine（`trtexec --fp16`）

### 阶段二：推理封装（2-3 天）

- [ ] 创建 `trt_detector.hpp` / `trt_detector.cpp`
- [ ] 创建 `trt_preprocess.cu`（letterbox kernel）
- [ ] 创建 `cuda_utils.hpp`（公共 check_error）
- [ ] 编写独立测试程序，验证推理结果与 Python 一致

### 阶段三：集成替换（1-2 天）

- [ ] 修改 `vision.hpp`（替换 darknet 头文件和成员变量）
- [ ] 修改 `vision.cpp`（start/run/stop 三个函数）
- [ ] 修改 `vision/CMakeLists.txt`（替换链接库）
- [ ] 修改 `imageproc/imageproc.hpp`（移除多余 darknet include）
- [ ] 修改 `color/color.cpp`（本地 check_error 替代 darknet/cuda.h）
- [ ] 修改 `config.conf`（模型路径配置）

### 阶段四：测试验证（2-3 天）

- [ ] 编译通过，解决链接问题
- [ ] 静态图片检测对比
- [ ] 实际运行验证（球识别 + 门柱定位）
- [ ] 性能测试（帧率 ≥ 20 FPS）
- [ ] 全流程测试（行走 + 识球 + 踢球）

### 预计总工期：6-10 天

---

## 附录 A：文件变更汇总清单

```
新增文件（4 个）：
  src/controller/player/vision/trt_detector.hpp    ← TensorRT 检测器头文件
  src/controller/player/vision/trt_detector.cpp    ← TensorRT 检测器实现
  src/controller/player/vision/trt_preprocess.cu   ← Letterbox CUDA kernel
  src/lib/cuda_utils.hpp                           ← 公共 CUDA 工具函数

新增模型文件（部署时添加）：
  src/data/algorithm/best.engine                   ← TensorRT 序列化引擎

修改文件（6 个）：
  src/controller/player/vision/vision.hpp          ← TRTDetector 替换 network
  src/controller/player/vision/vision.cpp          ← start/run/stop 改动
  src/controller/player/vision/CMakeLists.txt      ← nvinfer 替换 darknet
  src/lib/imageproc/imageproc.hpp                  ← 移除多余 darknet include
  src/lib/imageproc/color/color.cpp                ← check_error 来源替换
  src/data/config.conf                             ← 模型路径变更

不修改文件：
  src/lib/CMakeLists.txt                           ← 保留 darknet（auto_marker 需要）
  src/tools/auto_marker/main.cpp                   ← 暂不改动
  src/lib/darknet/*                                ← 保留原样
  其他所有文件                                       ← 完全不受影响
```

## 附录 B：性能参考数据

| 模型 | Orin NX FP16 | 参数量 | 推荐场景 |
|------|-------------|-------|---------|
| YOLOv8n (640) | ~3-5 ms | 3.2M | **首选**，满足实时性 |
| YOLOv8s (640) | ~6-10 ms | 11.2M | 精度优先 |
| YOLOv8m (640) | ~15-25 ms | 25.9M | 不推荐（可能不够实时） |
| 原 YOLOv3 (416) | ~10-15 ms | - | 当前基准 |

## 附录 C：关键注意事项速查

1. **类别 ID 必须一致**：训练 yaml 中 0=post, 1=ball，与代码 `ball_id_=1`, `post_id_=0` 对应
2. **Engine 不可跨平台**：必须在 Orin NX 上用 `trtexec` 生成，升级 TensorRT 后需重新生成
3. **Letterbox 坐标还原**：`real_x = (net_x - pad_x) / scale`，这是最易出 bug 的地方
4. **输入格式 CHW**：letterbox kernel 需要输出 CHW 排列，不是 HWC
5. **不修改 `object_det`**：所有下游模块依赖此结构，检测器输出必须兼容
6. **`check_error` 需本地定义**：移除 darknet 依赖后，color.cpp 和 vision.cpp 各需定义本地版本
7. **保留 darknet 子目录**：auto_marker 工具仍需要，暂不删除
