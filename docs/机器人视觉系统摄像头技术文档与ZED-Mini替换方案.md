# 机器人视觉系统摄像头技术文档与 ZED-Mini 替换方案

## 1. 文档目的与范围

本文档面向当前 7_26_V0 代码库，目标如下：

1. 说明当前 MindVision 单目摄像头在机器人系统中的功能实现与数据流。
2. 明确摄像头模块向其他模块输出的数据内容、格式、传输方式与更新频率。
3. 给出摄像头相关 API 与远程控制接口说明，并提供调用示例。
4. 制定将 MindVision 单目方案替换为 ZED-Mini 双目方案的可实施路线，覆盖 SDK 集成、代码修改、参数调整、性能优化、差异应对、测试验收。

## 1.1 对本次替换建议的校核结论

本次建议总体判断为：方向正确，可作为实施输入；但有若干点需要按当前代码现状修正后再执行。

正确且建议保留的要点：

1. 现有 Camera 模块确实是 MVSDK + V4L2 双后端。
2. MindVision 路径下确实关闭了自动曝光并支持曝光/增益硬件控制。
3. 采集与视觉处理是异步链路，Camera 在独立线程，Vision 在定时器线程。
4. 用 ZED 深度替代单目逆投影测距是正确升级方向。
5. GPU 零拷贝是 Orin 平台上达标帧率的关键优化项。

需要修正后再执行的点：

1. 当前 WorldModel 输出并非完整三维球坐标，主链路是二维地面坐标（self/global）+ 像素坐标。
2. 当前 vision_period 配置为 50ms，基线处理频率约 20Hz，而非 30Hz。
3. ZED SDK 的接入主目标应优先放在 sensor 模块（Camera 后端），而不是仅在 vision 目标中直接链接。
4. 不能立即彻底废弃 odometry；应先保留作为深度异常时的回退路径。
5. 深度取值不应只取检测框中心单点，需改为 ROI 中值/分位数融合以抑制噪声和空洞。

---

## 2. 当前 MindVision 摄像头实现总览

## 2.1 架构位置与生命周期

当前摄像头模块是 Player 的一个 Sensor 子类实例，由 Player 在注册阶段创建并启动。

- 创建与挂接：Player.regist 中创建 camera 传感器并 attach 到 Vision。
- 启动顺序：先启动 Camera，再启动 Vision。
- 关闭顺序：Player.unregist 中先停止 camera，再停止 Vision。

实现特点：

1. 统一 Camera 抽象，内部支持两条采集后端。
2. 首选 MindVision SDK（MVSDK）；若未枚举到设备，自动回退到 V4L2 USB 摄像头路径。
3. 采集完成后通过发布订阅 notify(SENSOR_CAMERA) 推送给 Vision。

## 2.2 实际具备的全部功能

当前 Camera 模块在代码中已实现如下能力。

1. 设备打开与自动回退。
- 首选 MVSDK：CameraEnumerateDevice -> CameraInit -> CameraPlay。
- 回退 V4L2：打开 image.dev_name（默认 /dev/video0），配置 YUYV + mmap 缓冲。

2. 图像采集线程。
- Camera.start 创建采集线程，循环拉取图像。
- MVSDK 路径使用 CameraGetImageBuffer，成功后通知并释放缓冲。
- V4L2 路径使用 VIDIOC_DQBUF/QBUF 完成环形队列采集。

3. 相机参数在线更新（MVSDK 路径）。
- 支持 exposure_gain（模拟增益）和 exposure_time（曝光时间）实时下发。
- 参数源来自 data/device/camera2.conf。
- 通过远程调试命令 CAMERA_SET 触发。

4. 分辨率与帧数据访问。
- 提供 camera_w、camera_h、camera_size、buffer 等访问接口。
- Vision 在 updata 回调中读取 buffer 并拷贝到本地缓存。

5. 时间观测字段（MVSDK）。
- Camera 维护 timestamp_begin、timestamp_end、time_used，便于观测取流耗时。

6. 视觉前处理输入适配。
- MVSDK 输出按 Bayer 流处理（cudaBayer2BGR）。
- V4L2 输出按 YUYV 流处理（cudaYUYV2BGR）。

7. 与视觉检测和定位链路联动。
- Vision 获取图像后执行缩放、去畸变、TensorRT 检测。
- 结果用于球位置更新、球门观测、定位更新和调试图像回传。

## 2.3 数据流拓扑

1. Camera 采集线程产生原始帧。
2. Camera 调用 notify(SENSOR_CAMERA)。
3. Vision.updata 收到回调并拷贝当前帧到 camera_src_。
4. Vision 定时器 run（vision_period）触发，执行 GPU 前处理与检测。
5. 输出流向：
- WorldModel：球可见性、像素位置、自身坐标/全局坐标。
- Localization：球门观测 posts_。
- 调试端（TCP）：JPEG 图像流、点选距离响应。

---

## 3. 摄像头模块输出数据说明

## 3.1 输出到 Vision 的原始帧数据

### 3.1.1 数据内容

- 指针：unsigned char*（来自 Camera.buffer()）。
- 宽高：camera_w, camera_h。
- 数据长度：camera_size。

### 3.1.2 数据格式

1. MVSDK 路径：
- 格式：单通道 Bayer 原始流。
- 大小：w * h 字节。

2. V4L2 路径：
- 格式：YUYV 4:2:2。
- 大小：w * h * 2 字节。

### 3.1.3 传输方式

- 进程内发布订阅（Publisher/Subscriber）。
- 无网络序列化，Vision 在回调中 memcpy 到 camera_src_。

### 3.1.4 更新频率

- Camera 通知频率跟随实际采集帧率。
- MVSDK 循环每次成功取帧后通知，循环末尾 usleep(1000)。
- V4L2 路径每次 DQBUF/QBUF 后通知，循环末尾 usleep(10000)。
- 上层 Vision 真正处理周期由 vision_period 控制，当前配置为 50ms，即约 20Hz。

## 3.2 输出到世界模型与定位模块的数据

### 3.2.1 球体目标结果（Vision -> WorldModel）

- 来源：TensorRT 检测结果 ball_dets_。
- 输出内容：
- 目标像素坐标（球框底部中心）。
- 里程计估计距离与方位变换后的 self/global 二维坐标。
- alpha, beta 归一化像素偏移。
- can_see 可见标志。

当连续超过约 500ms 未见球时，置球不可见。

### 3.2.2 球门观测（Vision -> Localization）

- 来源：post_dets_。
- 输出结构：goal_post 列表，包含类型、距离（cm）、方位角。
- 限制条件：距离大于 600cm 的候选被过滤。

## 3.3 输出到调试端的数据

### 3.3.1 图像流

- 命令类型：IMG_DATA。
- 载荷：JPEG 二进制（cv::imencode 结果）。
- 用途：调试界面实时显示 origin/result 画面。

### 3.3.2 点选距离响应

- 命令类型：REMOTE_DATA。
- 载荷结构：
- remote_data_type = IMAGE_SEND_TYPE
- image_send_type = IMAGE_SEND_DIS
- float x
- float y

- 语义：调试端点击图像点后，返回该点估计的地面坐标距离分量。

---

## 4. 摄像头相关 API 与接口说明

## 4.1 Camera 类核心接口

定义位置：src/controller/player/sensor/camera.hpp。

1. bool start()
- 功能：打开设备并启动采集线程。
- 返回：设备初始化与线程创建成功返回 true。

2. void stop()
- 功能：停止采集循环、关闭设备与缓冲。

3. bool open() / void close()
- 功能：底层设备资源打开/释放。
- 备注：open 内部自动选择 MVSDK 或 V4L2。

4. unsigned char* buffer() const
- 功能：返回当前帧指针。
- 备注：MVSDK 与 V4L2 的内存来源不同，调用方无需区分来源。

5. int camera_w() const / int camera_h() const / int camera_size() const
- 功能：返回当前相机分辨率与帧字节数。

6. bool use_mv() const
- 功能：指示当前是否使用 MindVision SDK 路径。

7. void set_camera_info(const camera_info& para)
- 功能：在线设置相机参数（当前仅 MVSDK 路径有效）。
- 当前支持：
- id=1 -> CameraSetAnalogGain
- id=2 -> CameraSetExposureTime

## 4.2 Vision 侧相关接口

定义位置：src/controller/player/vision/vision.hpp。

1. void updata(const pub_ptr& pub, const int& type)
- 处理 Camera 的 SENSOR_CAMERA 通知。
- 首次回调时初始化 GPU 缓冲与输入尺寸。

2. bool start() / void stop()
- start：加载 TensorRT 引擎，分配 CUDA 缓冲，启动周期定时器。
- stop：释放 detector 与 CUDA 资源。

3. void set_camera_info(const camera_info& para)
- 更新 Vision 内部 camera_infos_，用于颜色还原参数（饱和度与 RGB 增益）。

4. void get_point_dis(int x, int y)
- 输入像素点，返回估计距离并通过 REMOTE_DATA 回传。

## 4.3 远程控制接口（TCP）

协议定义：src/lib/tcp.hpp。

1. CAMERA_SET
- 方向：调试端 -> 机器人。
- 载荷：id(int) + value(float)。
- 作用：同时更新 Vision 参数缓存和 Camera 硬件参数。

2. IMAGE_SEND_TYPE
- 方向：调试端 -> 机器人（请求）或机器人 -> 调试端（响应）。
- 常用值：
- IMAGE_SEND_ORIGIN：发送原图模式。
- IMAGE_SEND_RESULT：发送检测叠加图模式。
- IMAGE_SEND_POINT：请求点距计算。
- IMAGE_SEND_DIS：返回点距结果。

## 4.4 使用示例

### 4.4.1 示例 A：Player 注册并启动摄像头

```cpp
sensors_["camera"] = std::make_shared<Camera>();
sensors_["camera"]->attach(VISION);
sensors_["camera"]->start();
if (!VISION->start()) {
    return false;
}
```

### 4.4.2 示例 B：远程设置曝光

```cpp
camera_info info;
info.id = 2;        // exposure_time
info.value = 8.0f;  // ms
VISION->set_camera_info(info);
auto cm = std::dynamic_pointer_cast<Camera>(get_sensor("camera"));
cm->set_camera_info(info);
```

### 4.4.3 示例 C：Vision 接收摄像头帧

```cpp
if (type == Sensor::SENSOR_CAMERA) {
    auto sptr = std::dynamic_pointer_cast<Camera>(pub);
    memcpy(camera_src_, sptr->buffer(), src_size_);
}
```

---

## 5. ZED-Mini 双目替换方案

## 5.1 目标与原则

替换目标：在不破坏现有行为控制、检测、调试链路的前提下，将图像源从 MindVision 单目迁移到 ZED-Mini 双目，并逐步引入深度能力。

实施原则：

1. 先兼容、后增强：第一阶段保证现有单目算法可直接跑通。
2. 保持接口稳定：尽量不改 Player/Vision 外部调用方式。
3. 可灰度切换：通过配置选择 MindVision、V4L2、ZED 后端。

## 5.2 SDK 集成方法

## 5.2.1 依赖安装

1. 安装 ZED SDK（与 JetPack/CUDA 版本匹配）。
2. 校验头文件与库可见：
- zed/Camera.hpp
- libsl_zed.so 及相关依赖

## 5.2.2 CMake 集成建议

在传感器模块引入 ZED 库，建议优先在 src/controller/player/sensor/CMakeLists.txt 增加后端依赖（因为 Camera 位于该模块）：

```cmake
option(USE_ZED "Enable ZED backend" ON)

if(USE_ZED)
  find_package(ZED REQUIRED)
  target_compile_definitions(sensor PRIVATE USE_ZED)
  target_include_directories(sensor PRIVATE ${ZED_INCLUDE_DIRS})
  target_link_libraries(sensor ${ZED_LIBRARIES})
endif()
```

注意事项：

1. x86_64 与 aarch64 的库路径不同，建议按 CROSS 分支分别配置。
2. 保留原 MVSDK 链接，过渡期支持双实现并行构建。
3. vision 模块仅在需要直接消费 ZED 类型时再链接 ZED；优先维持 Camera 对 Vision 的抽象隔离。

## 5.3 代码修改范围与改造路线

## 5.3.1 修改范围清单

核心改动文件建议如下：

1. src/controller/player/sensor/camera.hpp
- 引入后端枚举 CameraBackend。
- 新增 ZED 句柄与左右目/深度缓冲成员。

2. src/controller/player/sensor/camera.cpp
- open/run/close 增加 ZED 分支。
- 新增 ZED 参数初始化与 grab/retrieveImage 流程。

3. src/lib/model.hpp
- 扩展 camera_param，支持双目内参与基线（fx_l, fx_r, cx_l, cx_r, baseline 等）。

4. src/controller/player/vision/vision.hpp 与 vision.cpp
- 当前 use_mv_ 语义需升级为 pixel_format/backend 判断。
- 为 ZED 左目 BGR 直出路径增加零拷贝或最小拷贝逻辑。

5. src/data/config.conf
- 新增 camera.backend、camera.zed.resolution、camera.zed.fps、camera.zed.depth_mode 等。

6. src/data/device/*.conf 与 src/data/model/*.conf
- 拆分并新增 zed_left.conf、zed_right.conf 或统一 zed_camera.conf。

## 5.3.2 分阶段实施

阶段 1：等效替换（单目兼容）

1. ZED 仅输出左目图像到现有 Vision。
2. 保持 object_det 与 WorldModel 逻辑不变。
3. 调试端协议保持不变。

阶段 2：双目增强

1. 引入深度图/点云估距，新增 depth_odometry 路径而不是直接删除原 odometry。
2. 在 Vision 中增加基于深度与几何反投影的融合，建议优先采用 ROI 深度中值而非单点深度。
3. 建立异常回退：深度无效、空洞、强反光时回退到单目几何估计。

阶段 3：性能收敛与鲁棒性增强

1. 使用 ZED GPU Mat 到 CUDA 处理链的直接传递。
2. 引入掉帧监测与自动重连。
3. 完成多光照和高动态场景参数标定。

## 5.4 参数配置调整建议

建议新增配置项示例：

```json
"camera": {
  "backend": "zed",
  "zed": {
    "resolution": "VGA",
    "fps": 60,
    "depth_mode": "PERFORMANCE",
    "coordinate_units": "METER",
    "min_depth": 0.2,
    "max_depth": 12.0,
    "exposure": 40,
    "gain": 50,
    "whitebalance_auto": true
  }
}
```

调整重点：

1. 分辨率与检测输入尺寸联动，避免重复缩放。
2. FPS 与 vision_period 联动，建议 camera_fps >= 2 * 视觉处理频率。
3. 深度模式在性能与精度之间按平台分档（x86_64 与 Orin 分开配置）。

## 5.5 新旧摄像头功能差异与应对策略

| 维度 | MindVision 单目 | ZED-Mini 双目 | 应对策略 |
|---|---|---|---|
| 成像通道 | 单目 | 双目（左右）+ 深度 | 第一阶段仅接左目保证兼容，第二阶段引入深度融合 |
| 距离估计 | 几何反投影+IMU姿态 | 可直接深度测距 | 近距优先深度，远距保留几何估计并做融合 |
| 曝光增益控制 | MVSDK 参数接口 | ZED Video Settings | 建立统一 CameraControl 抽象层 |
| 像素格式 | Bayer 或 YUYV | BGR/RGBA（SDK 输出） | 在 Vision 增加格式分发，删除不必要转换 |
| 标定参数 | 单目内参+畸变 | 双目内外参+基线 | 扩展 camera_param 与标定文件结构 |
| 带宽/算力 | 较低 | 较高（尤其深度） | 降分辨率、限制深度范围、异步流水 |
| 依赖环境 | MVSDK/V4L2 | ZED SDK + CUDA | 增加环境自检与后端回退机制 |

## 5.6 性能优化建议

1. 内存与拷贝优化
- 尽量使用 GPU 侧图像句柄，减少 Host<->Device 复制。
- 采用 pinned memory 与 cuda stream 重叠拷贝和计算。
- ZED 路径优先使用 sl::Mat 的 GPU 内存指针直连预处理，避免 Device->Host->Device 往返。

2. 线程与队列优化
- 摄像头采集线程与视觉处理线程解耦，使用无锁队列或双缓冲。
- 丢弃过期帧，优先处理最新帧，降低控制闭环延迟。

3. 推理链路优化
- 结合 TensorRT batch=1 低延迟配置。
- 在 ZED 输出分辨率与网络输入尺寸之间做一次性缩放。

4. 运行稳定性
- 增加 grab 超时、设备断连重试、帧时间戳监控。
- 记录端到端延迟、FPS、丢帧率并定期上报。

5. 深度质量优化
- 对检测框区域执行深度去噪（中值/双边）与置信度过滤。
- 对球和球门分别设置有效深度窗口，避免极近/极远误值污染定位。

## 5.7 系统兼容性设计

1. 通过配置选择后端：mv / v4l2 / zed。
2. 保持 Camera 类对外接口不变，避免影响 Player 与 Vision 调用关系。
3. 远程调试协议兼容，CAMERA_SET 与 IMAGE_SEND_TYPE 不变。
4. 允许构建时裁剪：无 ZED SDK 环境下仍可编译原方案。

---

## 6. 测试验证流程与验收标准

## 6.1 测试流程

阶段 A：编译与启动验证

1. 完成 ZED SDK 链接后全量构建通过。
2. 后端配置为 zed 时可正常启动并持续取流。
3. 后端配置回 mv/v4l2 时行为不回归。

阶段 B：功能回归验证

1. 球与球门检测结果可正常输出。
2. 调试端 origin/result 图像显示正常。
3. CAMERA_SET 与 IMAGE_SEND_POINT/DIS 指令链路正常。

阶段 C：性能验证

1. 在目标平台采集 30 分钟连续运行。
2. 统计 FPS、端到端延迟、GPU 占用、丢帧率。
3. 异常工况（拔插、强光、低光）下恢复能力验证。

阶段 D：精度验证

1. 固定标定场景下比较球距离误差。
2. 定位场景下比较球门观测稳定性与漂移。
3. 与旧方案对比检测召回率和误检率变化。

## 6.2 建议验收标准

1. 功能完整性
- 现有单目功能 100% 可用：检测、调试图传、远程控制、定位输入。

2. 性能指标
- 视觉处理频率不低于当前基线（20Hz）。
- 端到端延迟较当前方案不劣化超过 10%。
- 连续运行 30 分钟无崩溃、无内存泄漏趋势。

建议增加目标档（可选）：

1. 若启用 GPU 零拷贝并完成链路优化，目标处理频率可提升到 30Hz。
2. 对应单帧视觉主循环目标耗时不超过 30ms。

3. 精度指标
- 球体测距中位误差不高于旧方案，或在 3m 内提升至少 15%。
- 球门观测稳定性（抖动标准差）不高于旧方案。

4. 兼容性指标
- 不改上层行为模块接口。
- 配置可切换且切换后可独立稳定运行。

---

## 7. 风险与缓解

1. SDK 版本与 CUDA/JetPack 不匹配
- 缓解：锁定版本矩阵，CI 增加环境检查。

2. 双目深度算力开销过高
- 缓解：先禁用深度或降低分辨率，分阶段启用。

3. 颜色/曝光在不同场地差异明显
- 缓解：保留在线参数调节通道，增加配置模板。

4. 多后端分支导致维护复杂度上升
- 缓解：抽象统一后端接口，避免在 Vision 层散落判断。

---

## 8. 结论

当前系统已具备完整的摄像头采集、视觉处理、调试交互闭环。推荐采用“后端可切换 + 分阶段迁移”策略替换为 ZED-Mini：

1. 第一阶段完成左目兼容替换，确保业务无感。
2. 第二阶段引入深度信息提升测距与定位鲁棒性。
3. 第三阶段完成性能优化与稳定性收敛，达到可比赛部署标准。

该路线可在控制风险的同时，最大化利用 ZED-Mini 双目能力并保持系统兼容性。