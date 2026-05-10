# 2026-03-23 auto_marker 升级工作日志（Darknet -> TensorRT10）

## 变更 1
- 时间：2026-03-23
- 文件：`src/tools/auto_marker/main.cpp`
- 内容：
  - 移除 Darknet 头文件与推理调用（`network_predict/get_network_boxes/do_nms_sort`）
  - 接入 `TRTDetector`（TensorRT10 name-based API 封装）
  - 新增命令行参数：`auto_marker <images_path> [engine_path]`
  - 默认引擎路径改为 `data/algorithm/best.engine`
  - 保持原标注语义：输出 `class 0=ball, class 1=post`
  - 保持原阈值与过滤规则：
    - ball: `prob>=0.5, min_w/h=20, |w/h-1|<0.3`
    - post: `prob>=0.4, min_w=15, min_h=20`
  - 输出框由像素框转换为 YOLO txt 归一化格式（cx cy w h）

## 变更 2
- 时间：2026-03-23
- 文件：`src/tools/auto_marker/CMakeLists.txt`
- 内容：
  - 去除 `darknet` 链接
  - 增加 `../../controller/player/vision/trt_detector.cpp` 编译源
  - 增加 `TRTDetector` 头文件目录
  - 增加 TensorRT/CUDA 链接：`nvinfer nvinfer_plugin cudart`

## 当前状态
- auto_marker 已完成从 Darknet 到 TensorRT10 的主流程迁移。
- 下一步：执行构建检查，修复可能的编译问题并补充日志。

## 变更 3
- 时间：2026-03-23
- 文件：`src/tools/auto_marker/main.cpp`、`src/tools/auto_marker/CMakeLists.txt`
- 内容：
  - 执行构建验证：`cmake -S . -B build && cmake --build build --target auto_marker -j4`
  - 构建结果：`Built target auto_marker`
  - 说明：当前仅存在 TensorRT 头文件的 deprecated 警告，不影响本次迁移功能
