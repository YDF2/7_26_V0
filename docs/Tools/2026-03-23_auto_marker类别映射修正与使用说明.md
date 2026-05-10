# 2026-03-23 auto_marker 类别映射修正与使用说明

## 背景

根据《视觉检测逻辑与类别 ID 一致性检查》文档，在线 Vision 使用类别映射为：

1. ball = 1
2. post = 0

而 auto_marker 之前默认使用 0/1，容易造成离线验证结论与在线行为不一致。

## 本次修改

修改文件：
1. src/tools/auto_marker/main.cpp

改动内容：
1. 默认类别映射改为与在线一致：ball_id=1、post_id=0。
2. 增加可选命令行参数：
- ball_id post_id
- ball_thresh post_thresh
3. 增加启动参数打印，明确本次运行的引擎、类别映射、阈值。
4. 增加结束统计打印：images、ok_frames、total_ball、total_post。

## 新用法

```bash
auto_marker <images_path> [engine_path] [ball_id post_id] [ball_thresh post_thresh]
```

示例：

1. 使用默认（在线一致 1/0）：
```bash
auto_marker imgs data/algorithm/best.engine
```

2. 显式指定 1/0 与阈值：
```bash
auto_marker imgs data/algorithm/best.engine 1 0 0.5 0.4
```

3. 若需对比旧数据映射，可切到 0/1：
```bash
auto_marker imgs data/algorithm/best.engine 0 1 0.5 0.4
```

## 预期收益

1. 离线验证与在线 Vision 的类别解释保持一致。
2. 快速定位“无框”是否来自 engine/阈值，而不是类别映射误差。
3. 通过统计量可直接判断模型召回是否异常。

## 校验

1. 已对修改文件执行 IDE 诊断检查，无错误。
