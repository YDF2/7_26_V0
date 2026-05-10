# 2026-03-24 视觉链路核查：debuger图像与TensorRT推理输入是否一致

## 结论（先给答案）

1. 传给debuger的图像 与 送去TensorRT推理的图像不是同一张“最终输入图”，但来源相同。
2. 两者共同来源都是 `dev_undis_`（处理分辨率为 `w_ x h_`，当前配置是 640x480）。
3. 送给TensorRT前会额外执行一次 resize 到 engine 输入尺寸 `in_w x in_h`，并执行 BGR->RGB + float 归一化。
4. 因此，是否与训练时 640x480 匹配，取决于 engine 输入尺寸是否也是 640x480。
5. 从仓库现有证据看，当前模型输入高概率是 640x640，不是 640x480。

## 代码依据

### 1) debuger发送图像来源

- Vision 主流程先生成 `dev_undis_`。
- debug发送分支从 `dev_undis_` 拷回主机 `bgr`，然后 `send_image()` 做 JPG 编码并通过 TCP 发送。

关键位置：
- `src/controller/player/vision/vision.cpp:207` 到 `vision.cpp:222`（`dev_undis_` 生成）
- `src/controller/player/vision/vision.cpp:346`、`vision.cpp:352`（从 `dev_undis_` 拷回）
- `src/controller/player/vision/vision.cpp:403` 到 `vision.cpp:414`（JPG编码并发送）

### 2) TensorRT推理输入来源

- 推理输入同样从 `dev_undis_` 出发。
- 先 resize：`dev_undis_ (w_ x h_) -> dev_sized_ (in_w x in_h)`。
- 再转为网络输入：`cudaBGR2RGBfp(dev_sized_, dev_rgbfp_, in_w, in_h)`。
- `detector_.detect(dev_rgbfp_, w_, h_, ...)` 入网推理。

关键位置：
- `src/controller/player/vision/vision.cpp:224` 到 `vision.cpp:227`
- `src/controller/player/vision/vision.cpp:229`

### 3) in_w / in_h 如何确定

- `in_w`、`in_h` 来自 TensorRT engine 的输入 tensor shape，不是固定写死在 Vision。
- `TRTDetector::load()` 中读取 `engine_->getTensorShape(input_name_)`，并赋值 `input_h_=d[2]`、`input_w_=d[3]`。

关键位置：
- `src/controller/player/vision/vision.cpp:560` 到 `vision.cpp:563`
- `src/controller/player/vision/trt_detector.cpp:90` 到 `trt_detector.cpp:94`

## 关于“训练是640x480，代码是否匹配”

### 配置侧

- 运行图像处理配置为 640x480：
  - `src/data/config.conf:28` (`image.width=640`)
  - `src/data/config.conf:29` (`image.height=480`)

### 模型侧

- 实际送推理尺寸由 engine 决定。


## 最终判断

1. debuger图像与推理图像“同源不同形态”：
   - 同源：都来自 `dev_undis_`。
   - 不同形态：推理前会 resize 到 engine 输入尺寸并转 RGB float；debug发送走 BGR/JPG（可能叠加框）。
2. 若训练时严格使用 640x480，而当前 engine 输入是 640x640，则当前推理输入不匹配训练尺寸。
3. 若希望严格匹配 640x480，需要确保导出ONNX和构建engine时输入即为 640x480（或采用与训练一致的预处理策略）。

## 建议的最小验证动作

1. 在 `TRTDetector::load()` 完成后打印一次 `input_w_`、`input_h_`。
2. 启动后核对日志中的 engine 输入尺寸，作为最终准确信息源。
3. 若日志显示 640x640，而训练为640x480，建议重新导出/构建与训练一致的模型输入尺寸。
