# ZED-Mini 单目接入快速指南（基于本地 zed-sdk）

## 1. 这份指南解决什么问题

目标：先把原 MindVision 相机替换为 ZED-Mini，但仅使用左目，按“单目”方式继续给 Vision 提供图像，尽量不改上层检测与定位逻辑。

---

## 2. 本地 zed-sdk 目录里哪些内容有用

你的本地目录在 zed-sdk，核心可参考内容如下：

1. tutorials/
- 作用：最小可运行示例，最快验证“能不能开机、能不能抓图”。
- 建议优先看：
  - tutorial 1 - hello ZED/cpp/main.cpp：最小 open/close。
  - tutorial 2 - image capture/cpp/main.cpp：grab + retrieveImage(VIEW::LEFT)。

2. camera control/
- 作用：曝光、增益、白平衡等视频参数控制示例。
- 用途：后续替代原 CameraSetExposureTime/CameraSetAnalogGain 的调参入口。

3. depth sensing/
- 作用：深度图、点云相关示例。
- 用途：你当前阶段可先不接入，仅保留后续升级入口。

4. object detection/
- 作用：ZED 自带目标检测示例。
- 用途：你现在用的是 TensorRT YOLO，不必替换；主要用来学习 SDK 图像/推理管线组织方式。

5. positional tracking/、spatial mapping/、body tracking/、global localization/
- 作用：定位建图人体跟踪等高级模块。
- 用途：和当前“先替换相机源”目标关系不大，可后看。

6. recording/
- 作用：SVO 录制/回放。
- 用途：离线复现视觉问题非常有价值，建议后续加入调试流程。

7. zed one/、virtual stereo/、sensors_api/
- 作用：其他设备形态与新接口示例。
- 用途：当前 ZED-Mini 左目替换阶段可暂不依赖。

8. zed-sdk/CMakeLists.txt
- 作用：总示例入口，告诉你有哪些 sample 子工程可编译。
- 重点：里面 add_subdirectory 的模块就是本地 SDK 示例清单。

---

## 3. 最小初始化流程（单目左目）

下面是最小流程，和你现有 Camera 设计一致：

1. 创建相机对象
- sl::Camera zed;

2. 设置初始化参数
- sl::InitParameters init_parameters;
- camera_resolution：建议先 VGA 或 AUTO（先保守跑通）。
- camera_fps：先 30。
- depth_mode：先 NONE（只当单目图像源）。
- coordinate_units：可设 METER。

3. 打开相机
- auto state = zed.open(init_parameters);
- 若失败，记录错误码并走你现有 V4L2 回退逻辑。

4. 循环抓图
- zed.grab();
- 成功后 zed.retrieveImage(image, sl::VIEW::LEFT, sl::MEM::CPU);

5. 格式适配到现有 Vision
- 若是 4 通道（BGRA）：转 BGR。
- 若是 3 通道：直接按 BGR 使用。
- 输出仍保持 unsigned char* buffer + w/h/camera_size 供 Vision::updata 使用。

6. 关闭释放
- stop/close 时调用 zed.close()，并释放本地 buffer。

---

## 4. 对你当前工程的接入要点

结合你当前 Camera 代码，建议保持以下策略：

1. 后端顺序
- MV 打不开 -> 尝试 ZED -> 再回退 V4L2。

2. 先只做左目单目
- Camera::run 中只 retrieveImage(VIEW::LEFT)。
- Vision 仍按 BGR 输入 TensorRT，不改 detector 接口。

3. 分辨率策略
- ZED 采集分辨率可和网络输入不同。
- 先保持 Vision 既有 image.width/image.height（如 640x480），在 Camera 或 Vision 中统一 resize。

4. 去畸变策略
- 当前阶段可以沿用现有流程。
- 若发现畸变模型不匹配，再单独引入 ZED 标定参数替换。

---

## 5. 最小可用代码骨架（便于对照）

```cpp
sl::Camera zed;
sl::InitParameters init_parameters;
init_parameters.depth_mode = sl::DEPTH_MODE::NONE;
init_parameters.coordinate_units = sl::UNIT::METER;
init_parameters.camera_resolution = sl::RESOLUTION::VGA;
init_parameters.camera_fps = 30;

auto state = zed.open(init_parameters);
if (state != sl::ERROR_CODE::SUCCESS) {
    // fallback to V4L2
}

sl::Mat zed_image;
while (is_alive_) {
    if (zed.grab() <= sl::ERROR_CODE::SUCCESS) {
        zed.retrieveImage(zed_image, sl::VIEW::LEFT, sl::MEM::CPU);
        // BGRA/BGR -> BGR buffer
        notify(SENSOR_CAMERA);
    }
}

zed.close();
```

---

## 6. 建议的本地验证顺序（10 分钟版）

1. 先单独跑 zed-sdk 的 tutorial 1，确认能 open/close。
2. 再跑 tutorial 2，确认 LEFT 图像稳定输出。
3. 回到主工程，仅替换 Camera 图像源为 ZED LEFT，保持 Vision/TensorRT 不动。
4. 观察日志：
- 相机初始化成功。
- Vision 可持续处理帧。
- detector 不报输入尺寸/格式错误。

如果以上四步都通过，就说明“MindVision -> ZED-Mini（单目左目）”替换已经跑通。
