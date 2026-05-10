# 2026-03-24 视觉推理 letterbox 对齐最小改造实施记录

## 工作目标

在不改动业务框架和阈值策略的前提下，按 ZED 官方 YOLO 示例的几何假设完成最小可用改造：

1. 推理前预处理从直接拉伸改为 letterbox。
2. 后处理坐标从简单比例缩放改为 letterbox 对称反算。
3. 输出检测框统一做图像边界 clamp，避免越界坐标传播。

## 改动范围

- src/lib/imageproc/imageproc.hpp
- src/lib/imageproc/imageproc.cu
- src/controller/player/vision/vision.hpp
- src/controller/player/vision/vision.cpp
- src/controller/player/vision/trt_detector.hpp
- src/controller/player/vision/trt_detector.cpp
- src/tools/auto_marker/main.cpp

## 实施内容

### 1) 图像预处理层新增 letterbox 接口

在 imageproc 中新增接口：

- `cudaResizeLetterbox(unsigned char* in, int iw, int ih, unsigned char* sized, int ow, int oh, float& scale, int& pad_x, int& pad_y)`

实现要点：

1. 计算 `scale = min(ow/iw, oh/ih)`。
2. 计算有效缩放尺寸 `new_w/new_h` 与 `pad_x/pad_y`。
3. 通过 CUDA kernel 在输出画布上完成：
   - ROI 内按双线性采样缩放原图。
   - ROI 外填充值 114（与常见 YOLO letterbox 一致）。

### 2) Vision 推理输入链路切换为 letterbox

`Vision::run()` 内原流程：

- `cudaResizePacked(dev_undis_, ... -> dev_sized_)`

改为：

- `cudaResizeLetterbox(dev_undis_, ... -> dev_sized_, letterbox_scale_, letterbox_pad_x_, letterbox_pad_y_)`

并把本帧 `scale/pad` 作为参数传给 `TRTDetector::detect()`。

### 3) TRT 后处理改为 letterbox 反算并 clamp

`TRTDetector::detect()` 新增入参：

- `float letterbox_scale`
- `int letterbox_pad_x`
- `int letterbox_pad_y`

坐标反算从“各向缩放比例”改为：

1. 先在模型输入空间从 `cx,cy,w,h` 得到 `x1,y1,x2,y2`。
2. 再做逆变换：`(coord - pad) / scale` 回到原图坐标。
3. 最后 clamp 到 `[0, orig_w-1]` 与 `[0, orig_h-1]`。
4. 对无效框（`w<=0` 或 `h<=0`）直接丢弃。

同时保留现有业务过滤逻辑不变：

- 球门/球阈值过滤
- 最小宽高过滤
- 球宽高比过滤
- 分类后 NMS 与排序

### 4) auto_marker 调用链兼容修复

由于 `TRTDetector::detect()` 增加了 letterbox 元信息参数，`auto_marker` 同步做了最小兼容：

1. 预处理改为 `cudaResizeLetterbox(...)`。
2. 把 `letterbox_scale/pad_x/pad_y` 传入 `detect()`。
3. 保持原有标注导出格式和阈值策略不变。

## 预期收益

1. 几何假设与官方示例一致，降低非方形输入下的框偏移风险。
2. 边界目标不再向下游传递负坐标/越界坐标。
3. 保持现有业务参数体系不变，回归风险可控。

## 验证说明

本次在编辑后执行了工作区问题检查，结果如下：

1. 业务改动文件未出现新的逻辑级错误提示。
2. 当前 Problems 面板仍存在本机环境相关的 includePath 提示（如 CUDA/OpenCV 头文件索引），属于工程索引配置问题，不是本次代码逻辑改动引入。

建议在目标设备上按现有构建流程做一次完整编译与运行验证，重点观察：

1. debug 图与检测框的边缘对齐情况。
2. 贴边目标是否仍出现负坐标或越界宽高。
3. 球/门检测数量级是否与改造前同量级。

## 本轮未纳入

为保持最小改动，本轮未包含以下项：

1. 输出解析动态类别数改造（去除 fields=6 假设）。
2. 预处理流到推理流的显式 event 同步。
3. TensorRT 日志级别可见性增强。

以上可作为下一轮增强项。