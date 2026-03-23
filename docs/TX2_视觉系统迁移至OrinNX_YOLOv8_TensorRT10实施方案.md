# TX2视觉系统迁移至OrinNX平台（YOLOv8 + TensorRT 10，FP16 Only）全新技术实施文档

## 1. 文档定位

本文是对既有两份迁移文档的合并重构版，目标是形成一份可直接执行、可直接验收的工程实施文档。与旧版相比，本版采用以下强约束：

- 仅使用 FP16，彻底移除 INT8 路径
- 完全舍弃 Darknet，不保留任何运行时与构建残留
- 以 TensorRT 10 Name-based API 为唯一实现标准
- 在关键链路增加高密度日志埋点，支持赛场快速排障

---

## 2. 当前系统复盘（基于现有代码）

## 2.1 视觉链路现状

当前 `Vision` 模块通过 `Timer` 周期执行 `run()`，并通过 `Subscriber` 接收传感器回调：

1. 相机线程进入 `Vision::updata(...SENSOR_CAMERA)` 写入 `camera_src_`
2. IMU线程进入 `Vision::updata(...SENSOR_IMU)` 更新姿态队列
3. 视觉定时线程执行 `Vision::run()`，完成预处理、检测、融合
4. 结果写入 `WM` 和 `SL`，供 `Player::run()/think()` 决策使用

## 2.2 关键模块关系

- `Player::regist()`：装配 Camera/IMU/Vision 观察者关系并启动线程
- `Vision::start()`：加载模型、分配GPU内存、启动视觉定时线程
- `Vision::run()`：检测主流程
- `WorldModel`：接收球和球门检测融合结果
- `SelfLocalization`：基于球门观测做定位修正

## 2.3 现状问题摘要（与迁移直接相关）

1. 存在初始化时序风险：`camera_src_` 可见早于 GPU 缓冲可用。
2. `is_busy_` 为普通 bool，存在并发竞态。
3. 预处理流与推理流可能不同步，存在撕裂风险。
4. 文档中存在 TensorRT 旧API描述，不适配 TRT10。
5. 旧方案保留 Darknet 分支，不满足“彻底剥离”目标。

---

## 3. 漏洞分析与修复决策

## 3.1 致命竞态：相机内存先可见导致抢跑

### 风险说明

如果 Camera 线程先看到 `camera_src_ != nullptr`，但 `dev_src_ / dev_bgr_` 尚未完成分配，`Vision::run()` 可能提前进入 `cudaMemcpy`，触发非法访问或崩溃。

### 修复策略（必须执行）

- 初始化顺序改为：
1. 读取相机参数
2. 先完成 `cudaMalloc(dev_src_)` 和 `cudaMalloc(dev_bgr_)`
3. 最后才 `camera_src_ = malloc(...)`
- 将 `is_busy_` 改为 `std::atomic_bool`
- 对一次性初始化增加 `LOG(LOG_INFO)` 与失败 `LOG(LOG_ERROR)`

## 3.2 异步流撕裂：预处理与推理跨流不同步

### 风险说明

`cudaUndistored` 在默认流执行，而 TRTDetector 若使用私有 stream_，则推理可能读取到尚未完成的图像。

### 修复策略（二选一，推荐方案A）

- 方案A：统一全流程单流（建议默认流，或统一自定义流）
- 方案B：事件同步
1. 预处理结束后 `cudaEventRecord(evt, preprocess_stream)`
2. 推理前 `cudaStreamWaitEvent(trt_stream, evt, 0)`

工程建议：首版以“单流+显式同步日志”上线，保证稳定后再做细粒度异步优化。

## 3.3 TRT10 API 兼容漏洞

### 风险说明

使用 `enqueueV2/getBindingDimensions` 会在 TRT10 编译失败或行为不一致。

### 修复策略（强制）

- 全部替换为 Name-based API：
- `getNbIOTensors`
- `getIOTensorName`
- `getTensorShape`
- `setTensorAddress`
- `enqueueV3`

## 3.4 Darknet 残留漏洞（架构层）

### 风险说明

任何 Darknet 头文件、库链接或工具残留，都会阻碍“纯TRT维护模型”目标，并增加依赖复杂度。

### 修复策略（强制）

- 删除 vision/imageproc/color 等路径中的 Darknet 直接依赖
- 从构建系统中移除 `add_subdirectory(darknet)`
- 如保留 PC 标注工具，改为 OpenCV DNN ONNX 路径，不再依赖 Darknet

---

## 4. 目标架构（OrinNX + YOLOv8 + TensorRT 10）

## 4.1 架构目标

- 平台：Jetson OrinNX
- 模型：YOLOv8（类别顺序固定：0=post, 1=ball）
- 推理：TensorRT 10 FP16
- 约束：保持上层行为与定位接口不变

## 4.2 分层设计

1. VisionInput
- 相机输入适配（Bayer/YUYV）
- 原始帧进入 GPU 缓冲

2. VisionPreprocess
- resize + undistort + letterbox
- BGR -> RGB
- 归一化 + HWC->CHW

3. TRTDetector
- engine 加载
- TRT10 Name-based 绑定
- enqueueV3 推理

4. VisionPostprocess
- YOLOv8 输出解析
- 坐标反算（反 letterbox）
- NMS 与阈值过滤
- 输出 `object_det`

5. VisionFusion
- `odometry/camera2self`
- `WM->set_ball_pos`
- `SL->update`

---

## 5. 模型转换与部署（FP16 Only）

## 5.1 ONNX 导出（PC）

```bash
yolo export model=best.pt format=onnx imgsz=640 opset=11 simplify=True dynamic=False
```

导出后必须校验：

- 输入 shape 固定为 `[1,3,640,640]`
- 输出 shape 与后处理逻辑一致（例如 `[1,6,8400]`）
- 类别顺序严格匹配 `post=0, ball=1`

## 5.2 Engine 构建（必须在 OrinNX 本机）

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=best.engine \
  --fp16 \
  --verbose
```

说明：

- 本方案不包含 INT8，不配置校准集，不维护双精度分支
- Engine 与设备架构/JetPack/TRT版本强绑定，不可跨平台复用

---

## 6. TensorRT 10实现规范（必须遵守）

## 6.1 初始化与维度解析

- 调用 `engine_->getNbIOTensors()` 枚举 I/O
- 通过 `getIOTensorName()` 获取输入输出张量名
- 使用 `getTensorShape(name)` 获取维度
- 将输入输出名称保存为成员变量，严禁硬编码索引绑定

## 6.2 推理执行

- `context_->setTensorAddress(input_name, dev_input)`
- `context_->setTensorAddress(output_name, dev_output)`
- `context_->enqueueV3(stream)`

## 6.3 内存策略

建议首版采用“稳定优先”策略：

- 输入：`cudaMalloc` 设备内存
- 输出：可选 pinned host 映射内存（`cudaHostAllocMapped`）
- 如启用零拷贝，必须验证：
- 输出带宽是否优于显式 D2H
- CPU 端读取时序是否与流同步一致

注：Orin 共享物理内存不等于“所有场景零拷贝都更快”，必须实测后固定。

---

## 7. 并发与线程调度规范

## 7.1 启动顺序（统一标准）

1. `Player::init()`
2. `Player::regist()`
3. Camera 启动
4. Vision 启动并完成模型与GPU初始化
5. IMU 启动
6. Player 主循环启动

## 7.2 共享数据访问规范

- `camera_src_`、`camera_matrix_` 访问必须受 `frame_mtx_` 保护
- IMU 队列访问必须受 `imu_mtx_` 保护
- `is_busy_` 使用 `std::atomic_bool`，并采用 compare_exchange 防重入

## 7.3 推荐防重入模式

```cpp
bool expected = false;
if (!is_busy_.compare_exchange_strong(expected, true)) {
    return;
}
// ... run body ...
is_busy_.store(false);
```

## 7.4 初始化安全模板

```cpp
if (camera_src_ == nullptr) {
    // 1) 读取参数
    // 2) cudaMalloc(dev_src_, dev_bgr_)
    // 3) 最后 camera_src_ = malloc(...)
}
```

---

## 8. 日志体系设计（高密度可观测性）

## 8.1 日志分层

- `LOG_ERROR`：初始化失败、推理失败、维度异常、内存分配失败
- `LOG_WARN`：帧丢弃、耗时超阈值、输入空帧、NMS异常值
- `LOG_INFO`：启动参数、模型信息、各阶段耗时、检测统计

## 8.2 必打日志点（最低要求）

1. 引擎加载
- engine路径、文件大小、TRT版本、输入输出shape

2. GPU内存
- 每个缓冲的分配结果与字节数

3. 每帧流程
- 预处理耗时
- 推理耗时
- 后处理耗时
- 总耗时与FPS

4. 业务结果
- ball/post检测数量
- 最高置信度
- 被阈值过滤数量

5. 异常与恢复
- 连续失败帧计数
- 自动降级或重试动作

## 8.3 建议日志格式

```text
[Vision][Frame=12345] preprocess=1.6ms infer=3.8ms post=0.9ms total=6.7ms ball=1 post=2
[TRTDetector][Init] input=images[1x3x640x640] output=output0[1x6x8400] fp16=on
```

---

## 9. CMake与依赖清理（彻底去Darknet）

## 9.1 必改项

1. 删除/禁用：`add_subdirectory(darknet)`
2. vision目标链接中移除 darknet
3. 替换为 TensorRT 链接：
- `nvinfer`
- `nvinfer_plugin`
- `cudart`

## 9.2 头文件与符号清理

- 删除 `#include "darknet/network.h"`
- 删除 `#include "darknet/parser.h"`
- 删除 `#include "darknet/cuda.h"`
- 对历史 `check_error` 调用：引入本地 `static` 版本或统一 `cuda_utils.hpp`

## 9.3 工具链策略

若 `auto_marker` 需要保留：

- 改为 OpenCV DNN ONNX 实现
- 不再通过 Darknet 编译任何目标

---

## 10. 关键代码改造清单

## 10.1 文件级改造

1. vision.hpp
- `bool is_busy_` -> `std::atomic_bool is_busy_`
- `network net_` -> `TRTDetector detector_`

2. vision.cpp
- 删除 Darknet 模型加载与推理代码
- `start()` 改为加载 `best.engine`
- `run()` 改为 `detector_.detect(...)`
- 修正 camera 初始化顺序

3. 新增 trt_detector.hpp/.cpp
- TRT10 Name-based API
- preprocess/postprocess/nms

4. 新增 trt_preprocess.cu
- letterbox + 归一化 + 通道变换

5. imageproc/color/CMake
- 清理 Darknet 依赖
- 补全 TensorRT 链接与 CUDA 源编译

## 10.2 代码审查必检项

- 不允许出现 `enqueueV2/getBindingDimensions`
- 不允许出现 `net_cfg_file/net_weights_file`
- 不允许出现 Darknet include 或 target_link_libraries(darknet)
- 检查 `camera_src_` 赋值位置是否在全部 GPU 分配之后
- 检查所有错误路径是否有 `LOG_ERROR`

---

## 11. 测试验证方案

## 11.1 功能验证

- 球/球门检出正确率
- 坐标反算准确性（含 letterbox 还原）
- 世界模型写入一致性
- 定位模块触发正确性

## 11.2 性能验证（FP16目标）

目标建议：

- 纯推理 3~6ms（按模型大小浮动）
- 单帧端到端 < 20ms
- 稳态 FPS >= 45（按相机帧率约束）

## 11.3 稳定性验证

- 连续运行 2 小时无崩溃
- 启停 100 次无资源泄漏
- 相机断连/重连可恢复
- Debug图像开关不会引发死锁

## 11.4 日志验收

- 每帧日志可定位耗时瓶颈
- 初始化失败日志可定位到具体步骤
- 阈值导致的漏检可通过日志复盘

---

## 12. 实施排期（建议）

## 阶段A：架构清理（1-2天）

- 删除 Darknet 依赖
- 建立 TRTDetector 骨架
- 完成 CMake 可编译

## 阶段B：推理打通（2-3天）

- 接入 engine
- 跑通 preprocess/enqueueV3/postprocess
- 输出 ball/post

## 阶段C：并发与日志（1-2天）

- 修复初始化竞态
- 统一流同步
- 增加全链路日志

## 阶段D：联调与压测（2-3天）

- WM/SL/Player 回归
- 性能调优
- 稳定性与容错测试

---

## 13. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| TRT10 API误用 | 无法编译或运行异常 | 统一Name-based API与代码审查门禁 |
| 流同步错误 | 随机误检、图像撕裂 | 单流策略或事件同步，禁止隐式跨流 |
| 类别映射错误 | 球门与球识别颠倒 | 强制校验 names 与 id 映射 |
| 竞态未清理 | 现场崩溃 | 原子防重入 + 初始化顺序修复 |
| 日志不足 | 故障定位慢 | 关键路径全部埋点并统一格式 |

---

## 14. 最终结论

本方案将迁移目标明确为“FP16单路线 + TensorRT10唯一后端 + Darknet彻底剥离 + 高可观测日志体系”。在此约束下：

1. 可以显著降低维护复杂度与线上不确定性。
2. 可以避免 TRT10 API 代际问题导致的编译失败。
3. 可以最大化赛场排障效率，确保问题可定位、可复现、可修复。

执行本方案后，视觉系统将从“可运行”升级为“可长期稳定运维”的工程级实现。
