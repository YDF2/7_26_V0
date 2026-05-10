# 7_26_V0 视觉系统迁移计划：YOLOv8 + TensorRT 10 + ZED-mini

## 版本信息
- 目标日期：2026-03-23
- 迁移目标：把 `Vision` 的检测器从 YOLOv3(Darknet) 替换为 YOLOv8 + TensorRT 10；把 `Camera` 的相机来源从 MindVision/V4L2 替换为 ZED-mini
- 第一阶段策略：使用 ZED-mini 左目图像作为单目输入，测距仍沿用现有 `Vision::odometry()`（不直接用 ZED 深度）

## 1. 目标与不变约束
### 1.1 必达目标
1. `Vision` 检测输出格式保持不变：仍产出 `std::vector<object_det>`，并继续驱动：
   - `WM->set_ball_pos(...)`（球）
   - `SL->update(posts_)`（门柱）
2. 相机采集到 `Vision` 的调用链不改动上层架构：
   - `Player` 仍通过 `sensors_["camera"]->attach(VISION)` 与 `Vision` 建立订阅关系
   - `Camera` 仍通过 `notify(SENSOR_CAMERA)` 把帧交给 `Vision::updata(...)`

### 1.2 不变约束（影响改造边界）
- 不改变 `object_det` / `goal_post` 等下游数据结构（`src/lib/model.hpp`）
- 不改变 `Player`/`FSM` 决策侧的输入来源语义

## 2. 当前系统架构与线程调用链（需要在实现时保持）
### 2.1 线程/调用链（现状）
```mermaid
flowchart LR
  Cam[Camera线程/采集循环] -->|notify(SENSOR_CAMERA)| VisionUpdata[Vision::updata(...SENSOR_CAMERA)]
  VisionUpdata -->|写 camera_src_ (frame_mtx_保护)| VisionTimer[Vision定时线程: Vision::run()]
  VisionTimer -->|预处理 + 推理 + 后处理| DetLogic[检测解析 -> ball_dets/post_dets]
  DetLogic --> WM[WM->set_ball_pos()]
  DetLogic --> SL[SL->update(posts)]
```

### 2.2 关键实现点（当前代码）
1. `Camera`（`src/controller/player/sensor/camera.hpp/.cpp`）
   - 内部采集循环调用 `notify(SENSOR_CAMERA)`
   - `Camera::buffer()` 返回当前帧指针
2. `Vision`（`src/controller/player/vision/vision.hpp/.cpp`）
   - `Vision::updata(...)`：收到 `SENSOR_CAMERA` 时把 `sptr->buffer()` `memcpy` 到 `camera_src_`
   - `Vision::run()`：周期执行
     - `cudaMemcpy(dev_src_, camera_src_, ...)` 把图像拷贝到 GPU
     - 根据相机输入类型走 `cudaBayer2BGR` 或 `cudaYUYV2BGR`
     - `cudaResizePacked` / `cudaUndistored` 做预处理
     - 调用 Darknet 推理（当前为 YOLOv3）
     - 解析 `detection` 并分流到 `ball_dets_` / `post_dets_`
3. `Player`（`src/controller/player/player.cpp`）
   - 注册阶段创建 `Camera` 并 attach 到 `VISION`
   - 启动顺序：`camera->start()` 之后调用 `VISION->start()`

### 2.3 并发风险与计划中的硬性修复
- 当前 `Vision` 里 `is_busy_` 是普通 `bool`，存在定时线程并发竞态风险
- 迁移实施时建议改为 `std::atomic_bool`，并使用 CAS 防止重复进入 `Vision::run()`（参考文档 `TX2_视觉系统迁移至OrinNX_YOLOv8_TensorRT10实施方案.md`）

## 3. 迁移总览（先做能跑通，再逐步增强）
### 3.1 分阶段路线
1. 阶段 1：ZED-mini 左目接入（单目），保持测距逻辑不变
   - `Camera` 增加 `ZED` 后端，向 `Vision` 输出能够匹配其预处理链路的像素格式
   - `Vision` 适配 ZED 输出（第一阶段建议走 BGR uint8 输入分支，减少新增 kernel）
2. 阶段 2：检测器替换为 YOLOv8 + TensorRT 10
   - 新增 `TRTDetector` 封装类，替换 `Vision::start/run/stop` 中 Darknet 推理逻辑
   - 解析结果仍填充 `object_det`，阈值/筛选规则保持一致（以便下游行为不变）
3. 阶段 3：联调与性能/鲁棒性收敛
   - 端到端延迟、帧率、检测数量统计与误检漏检回归
   - 引擎/坐标还原/内参匹配校验

## 4. 相机迁移：MindVision/V4L2 -> ZED-mini（第一阶段单目）
### 4.1 ZED-mini 接入方式
建议把 ZED SDK 集成优先放在 sensor 模块（`src/controller/player/sensor/`），让 `Vision` 只关心 `buffer()` 提供的图像数据。

根据你本地 ZED SDK samples（`zed-sdk/tutorials`）：
- 左目图像获取典型流程：
  - `zed.grab()`
  - `zed.retrieveImage(image, VIEW::LEFT, ...)`

ZED SDK 安装路径（本机已定位）：
- include：`/usr/local/zed/include`
- lib：`/usr/local/zed/lib/libsl_zed.so`

### 4.2 目标：让 `Vision::run()` 能正确消费输入
现有 `Vision::run()` 的分支逻辑是：
- `use_mv_ == true`：Bayer -> `cudaBayer2BGR`
- `use_mv_ == false`：YUYV -> `cudaYUYV2BGR`

因此第一阶段必须做两件事之一：
1. 让 ZED 输出落在现有分支之一（Bayer/YUYV），从而 `Vision` 无需改预处理
2. 或者更直接：扩展 `Vision`，让其支持 ZED 左目 BGR uint8 输入，并在预处理阶段跳过 Bayer/YUYV kernel

本计划采用方案 2（更可控，第一阶段更少不确定转换）：
- `Camera(ZED)` 输出 `BGR uint8` HWC packed（每帧 size = `w*h*3`）
- `Vision` 增加一个输入分支：当输入是 `BGR uint8` 时直接走
  - `cudaResizePacked(dev_bgr_, w_, h_, ...)` 并跳过 `cudaBayer2BGR/cudaYUYV2BGR`

### 4.3 标定/坐标一致性要求（第一阶段测距仍用 odometry）
你选择了第一阶段不使用深度，因此距离仍由 `Vision::odometry()` 的逆投影完成。
这意味着：
- `Vision` 当前使用的相机内参/畸变参数必须与 ZED 输出图像的几何关系一致
- 需要重点核对（或后续重新标定/映射）：
  - `data/model/camera2.conf`：`fx, fy, cx, cy, k1, k2, p1, p2`

## 5. 检测器迁移：YOLOv3(Darknet) -> YOLOv8 + TensorRT 10
### 5.1 关键约束（保持 `Vision` 输出行为不变）
`Vision::run()` 当前做了两类事：
1. 将输入图像预处理到 darknet 输入形态
2. 推理后解析 `detection` 并填充 `ball_dets_/post_dets_`
3. 业务侧融合逻辑从 `ball_dets_/post_dets_` 出发（保留）

迁移要求：
- 仅替换“推理与解析”部分
- 维持球/门柱的筛选阈值、宽高比过滤逻辑，以及 `ball_id_=1` / `post_id_=0` 的类别映射方式

### 5.2 TensorRT 10 接入设计（建议命名封装）
新增模块：
- `src/controller/player/vision/trt_detector.hpp`
- `src/controller/player/vision/trt_detector.cpp`
- （如需要 GPU 预处理）`src/controller/player/vision/trt_preprocess.cu`

目标类职责：
1. `load(engine_path)`：加载 `.engine`，初始化 TRT10 的 runtime/engine/context，并保存输入输出 tensor name 与形状
2. `detect(dev_bgr, img_w, img_h, ...)`：
   - letterbox 预处理（保持 YOLOv8 推荐流程，减少框偏差）
   - `enqueueV3` 推理
   - YOLOv8 输出解析 + NMS
   - 反 letterbox 坐标到原图像素空间
   - 以 `object_det` 填充 `ball_dets`/`post_dets`
3. `release()`：释放 TRT 与 CUDA 资源

### 5.3 TRT10 Name-based API 强约束
你已有专门的 TRT10 实施文档强调：
- 禁止使用旧 API：`enqueueV2/getBindingDimensions` 等
- 必须改用 TRT10 的 name-based API：
  - `getNbIOTensors`
  - `getIOTensorName`
  - `getTensorShape`
  - `setTensorAddress`
  - `enqueueV3`

### 5.4 视觉代码改造点（落地文件范围）
预计要改动的核心文件：
1. `src/controller/player/vision/vision.hpp`
   - 移除 Darknet 相关成员（`network net_`）
   - 引入 `TRTDetector detector_`
2. `src/controller/player/vision/vision.cpp`
   - `start()`：加载 `net_engine_file`
   - `run()`：把 Darknet 的推理与解析替换为 `detector_.detect(...)`
   - `stop()`：释放 TRT 资源
3. `src/controller/player/vision/CMakeLists.txt`
   - `darknet` 链接替换为 TensorRT：
     - `nvinfer`、`nvinfer_plugin`、`cudart`

4. `src/lib/imageproc/imageproc.hpp`
   - 当前包含了 `darknet/network.h`（虽然未使用），建议清理以减少 Darknet 依赖
5. `src/lib/imageproc/color/color.cpp`
   - 该文件目前可能依赖 darknet 的 `check_error`，迁移建议使用本地 `check_error`（例如已有 `src/lib/cuda_utils.hpp`）

## 6. 配置修改计划
### 6.1 模型引擎路径
当前 `src/data/config.conf` 已包含：
- `net_engine_file`
- `net_names_file`
但 `vision.cpp` 当前仍读 `net_cfg_file/net_weights_file`。

迁移实施时需要统一配置读取逻辑：
- `Vision::start()` 仅使用 `net_engine_file`，移除 darknet 配置项读取

### 6.2 相机后端配置（建议新增）
现有 `config.conf` 的 camera 配置较少（仅 `image.dev_name/width/height`）。
为支持后端切换，建议新增（示例）：
- `camera.backend = zed | mv | v4l2`
- `camera.zed.resolution`
- `camera.zed.fps`
- `camera.zed.depth_mode`（第一阶段可置 `NONE` 或不启用深度）

## 7. 测试与验收（建议按“对齐点”逐层验证）
### 7.1 阶段 1 验证：ZED 接入与输入格式正确性
1. 调试端能持续拿到 `origin/result` 图像（确认像素流稳定）
2. 校验 `Vision` 输入前处理阶段耗时与稳定性
3. 在相机未接入检测器（或检测器输出忽略）时，先确认：
   - `Vision::run()` 不崩溃
   - `dev_undis_` 结果正常（可通过调试图开关验证）

### 7.2 阶段 2 验证：TRTDetector 输出与 `object_det` 语义一致
1. 对比同一张图片的检测框：
   - Python/ONNX 推理（或已知正确引擎推理）
   - C++ TRTDetector 输出（ball/post box 的坐标与尺寸）
2. 类别映射一致性：
   - `post_id_ = 0`，`ball_id_ = 1`
3. `object_det` 基于阈值筛选后数量统计与排序逻辑一致

### 7.3 阶段 3 验证：端到端业务回归
1. 球可见时：
   - `WM->set_ball_pos()` 更新触发频率符合预期
   - 球不可见超过约 500ms 时置不可见逻辑仍有效
2. 门柱观测：
   - `SL->update(posts_)` 的更新次数与门柱稳定性达标
3. 性能指标（以你当前基线为准）：
   - `vision_period=50ms` 下主循环稳定（目标不低于约 20Hz）
   - 如发生掉帧，先检查相机端 fps 与 Vision 端耗时

## 8. 风险清单与排障要点
### 8.1 坐标偏差（框位置不准）
最可能原因：
- letterbox 与反算坐标映射错误
- ZED 输出几何与现有相机内参不一致
排障方式：
- 在 TRTDetector 内输出 `scale/pad_x/pad_y` 并对比 Python 实现
- 校验 `data/model/camera2.conf` 是否适配当前图像流

### 8.2 类别反了（球当门柱）
原因：
- YOLOv8 训练类别顺序与代码 `ball_id_/post_id_` 不一致
排障方式：
- 强制用同一套 yaml：`0=post, 1=ball`
- 在 postprocess 中记录 `prob_ball/prob_post` 进行快速定位

### 8.3 输入格式错配导致崩溃或全零检测
原因：
- Camera 输出的 buffer size 与 Vision 端 `camera_size_` 不一致
- Vision 端选择了错误的预处理分支
排障方式：
- 在 `Camera::buffer()/camera_size()` 与 `Vision::updata()` 处打印关键尺寸

### 8.4 并发竞态（间歇性崩溃）
原因：
- `Vision::run()` 与初始化/拷贝竞争
缓解：
- 使用 `std::atomic_bool` + CAS 防重入
- 强制初始化顺序（先完成 GPU malloc，再让 camera_src_ 可见）

## 9. 里程碑与时间线（建议）
1. M1（1-2 天）：ZED 后端接入 + Vision 输入格式适配（阶段 1 可跑）
2. M2（2-3 天）：实现 TRTDetector（阶段 2 可单测/静态回放）
3. M3（1-2 天）：联调检测 + 坐标/阈值回归
4. M4（2-3 天）：端到端 `WM/SL/Player` 回归 + 性能收敛

## 10. 交付物
1. 可运行版本：
   - ZED-mini 图像流进入 Vision
   - YOLOv8 TensorRT 10 输出 `ball_dets_/post_dets_`
2. 验收记录：
   - 检测框与业务坐标回归截图/统计（按测试计划）
3. 代码结构清理：
   - Darknet 运行时依赖最小化（如果还需保留仅用于标注工具）

