# 2026-03-23 ZED Streaming 与 Qt 调试方案完善日志

## 背景

针对“x86 debuger 黑屏”与“是否可直接采用 zed-sdk 官方 camera streaming 显示含检测框画面”问题，补充方案文档。

## 本次完成

1. 阅读并对照以下代码：
- zed-sdk/camera streaming/single_sender/cpp/src/main.cpp
- zed-sdk/camera streaming/receiver/cpp/src/main.cpp
- src/tools/debuger/image_monitor/image_monitor.cpp
- src/controller/player/vision/vision.cpp

2. 更新文档：
- docs/2026-03-23_x86_debuger_Qt黑屏原因分析与改进方案.md

3. 新增内容：
- 官方 streaming 是否可直接替换当前 debuger 的结论。
- 为什么官方 streaming 不直接携带 Vision 检测框。
- 面向工程落地的双通道方案（主业务流 + 原始流诊断）。
- 短期/中期/长期执行建议。

## 结论摘要

1. 官方 streaming 适合看原始画面，不适合直接替代当前“含检测框业务图 + REMOTE_DATA 交互”的主链路。
2. 推荐并行接入官方 streaming 作为辅助诊断通道，不替换主业务通道。
