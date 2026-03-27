# 2026-03-23 auto_marker 与 Qt 无框关联分析及代码补充

## 结论

1. auto_marker 不参与 controller 运行链路，不会直接导致 debuger Qt 无检测框。
2. 但 auto_marker 的默认类别映射与 Vision 可能不一致，会影响你对 engine 是否可用的判断，从而间接影响排障方向。

## 证据

1. auto_marker 仅在 tools 子工程中单独构建为可执行程序。
2. controller 运行链路没有调用 auto_marker。
3. Vision 里类别映射固定为 ball_id=1, post_id=0，而 auto_marker 之前默认是 load(engine, 0, 1)。

## 本次代码补充

修改文件：
1. src/tools/auto_marker/main.cpp

新增能力：
1. 支持命令行覆盖类别映射：
- ball_id
- post_id

2. 支持命令行覆盖阈值：
- ball_thresh
- post_thresh

3. 启动时打印当前参数（engine/类别映射/阈值）。
4. 运行结束打印统计（图片数、成功帧、球框总数、门柱框总数）。

## 新用法

```bash
auto_marker <images_path> [engine_path] [ball_id post_id] [ball_thresh post_thresh]
```

示例（与当前 Vision 映射一致）：

```bash
auto_marker imgs data/algorithm/best.engine 1 0 0.6 0.3
```

## 为什么这有帮助

通过和 Vision 完全一致的类别映射与阈值离线跑 auto_marker，你可以快速判断：

1. engine 在当前数据域是否有召回。
2. 是模型/参数问题，还是在线链路问题。

若离线统计长期接近 0，而 Qt 也无框，则优先处理 engine 版本、类别顺序和阈值。
