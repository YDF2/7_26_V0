# 机器人视觉系统工作日志索引
> 文档目的：系统性梳理docs目录中的工作日志，为后续agent提供快速定位和查询历史文档的导向性索引

---

## 目录

- [已完成工作](#已完成工作)
  - [ZED-Mini单目接入](#zed-mini单目接入)
  - [视觉检测逻辑与类别映射修正](#视觉检测逻辑与类别映射修正)
  - [Qt黑屏问题诊断与修复](#qt黑屏问题诊断与修复)
  - [auto_marker工具升级](#auto_marker工具升级)
  - [YOLOv8 TensorRT迁移方案设计](#yolov8-tensorrt迁移方案设计)
  - [视觉图像链路一致性核查](#视觉图像链路一致性核查)
  - [视觉推理letterbox最小改造](#视觉推理letterbox最小改造)
  - [视觉系统与ZED官方实现对标及去畸变集成](#视觉系统与zed官方实现对标及去畸变集成)
- [未完成工作](#未完成工作)
  - [TensorRT 10完整实施](#tensorrt-10完整实施)
  - [深度信息融合](#深度信息融合)
  - [性能优化与稳定性增强](#性能优化与稳定性增强)
- [技术文档索引](#技术文档索引)

---

## 已完成工作

### ZED-Mini单目接入

#### 工作描述
完成从MindVision单目摄像头到ZED-Mini左目单目的替换，实现相机后端的可切换架构。主要工作包括：

1. **Camera模块改造**：增加ZED后端支持，实现工厂标定参数读取
2. **Vision模块适配**：应用ZED左目标定参数，保持单目检测逻辑不变
3. **回退机制**：保持MV → ZED → V4L2的三级回退链路
4. **快速接入指南**：产出精简的接入文档和代码改造日志

#### 相关文档
- [ZED-Mini单目接入快速指南.md](file:///home/seurobot2/Downloads/7_26_V0/docs/ZED-Mini单目接入快速指南.md) - 最小可用流程说明
- [2026-03-23_ZED-Mini单目接入代码改造日志.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_ZED-Mini单目接入代码改造日志.md) - 具体代码改动记录
- [2026-03-23_ZED-Mini单目接入指南_工作日志.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_ZED-Mini单目接入指南_工作日志.md) - 文档产出说明
- [机器人视觉系统摄像头技术文档与ZED-Mini替换方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/机器人视觉系统摄像头技术文档与ZED-Mini替换方案.md) - 完整技术方案

#### 状态说明
✅ **已完成** - 代码改造完成，编译通过，具备运行条件

---

### 视觉检测逻辑与类别映射修正

#### 工作描述
解决在线Vision与离线auto_marker工具之间类别映射不一致的问题，确保检测结果的正确解释：

1. **类别映射统一**：确认数据集定义为ball=1, post=0
2. **在线检测验证**：检查Vision和TRTDetector的类别映射逻辑正确性
3. **离线工具修正**：修改auto_marker默认映射，增加命令行参数支持
4. **一致性检查**：识别并标记潜在风险点

#### 相关文档
- [2026-03-23_视觉检测逻辑与类别ID一致性检查.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_视觉检测逻辑与类别ID一致性检查.md) - 完整检查报告
- [2026-03-23_auto_marker类别映射修正与使用说明.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_auto_marker类别映射修正与使用说明.md) - 工具修改说明
- [2026-03-23_auto_marker与Qt无框关联分析及代码补充.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_auto_marker与Qt无框关联分析及代码补充.md) - 关联性分析

#### 状态说明
✅ **已完成** - 类别映射已统一，工具已升级

---

### Qt黑屏问题诊断与修复

#### 工作描述
诊断并修复ZED-Mini接入后x86端debuger Qt界面黑屏问题：

1. **根因定位**：确认问题出在ZED路径的去畸变remap阶段，映射越界导致大面积无效像素
2. **阶段A最小修复**：在ZED路径跳过去畸变，直接拷贝图像，快速恢复画面可见性
3. **Streaming方案评估**：分析ZED SDK官方camera streaming的适用性，确定不直接替换主业务链路
4. **调试方案完善**：制定双通道调试架构（主业务流 + 原始流诊断）

#### 相关文档
- [2026-03-23_x86_debuger_Qt黑屏原因分析与改进方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_x86_debuger_Qt黑屏原因分析与改进方案.md) - 完整问题分析与改进方案
- [2026-03-23_阶段A最小修复_ZED跳过去畸变实施记录.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_阶段A最小修复_ZED跳过去畸变实施记录.md) - 代码实施记录
- [2026-03-23_ZED_Streaming与Qt调试方案完善日志.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_ZED_Streaming与Qt调试方案完善日志.md) - Streaming方案分析

#### 状态说明
✅ **已完成** - 阶段A修复已实施，画面可见性已恢复

---

### auto_marker工具升级

#### 工作描述
完成auto_marker工具从Darknet到TensorRT 10的迁移，支持新的推理后端：

1. **Darknet依赖移除**：移除Darknet头文件和推理调用
2. **TensorRT 10集成**：接入TRTDetector，使用name-based API
3. **参数扩展**：增加命令行参数支持类别映射和阈值配置
4. **构建验证**：完成编译检查，修复潜在问题

#### 相关文档
- [2026-03-23_auto_marker_trt10_upgrade_log.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_auto_marker_trt10_upgrade_log.md) - 升级变更记录

#### 状态说明
✅ **已完成** - 工具已升级，构建通过

---

### YOLOv8 TensorRT迁移方案设计

#### 工作描述
设计从YOLOv3 Darknet到YOLOv8 TensorRT的完整迁移方案，保持原代码架构和下游模块不变：

1. **架构分析**：详细分析Darknet调用链和现有预处理流水线
2. **推理后端选择**：确定TensorRT C++ API为唯一方案
3. **模型转换流程**：定义ONNX导出和TensorRT Engine转换步骤
4. **代码改造方案**：提供完整的文件级改造清单和实施计划
5. **预处理差异详解**：说明YOLOv3和YOLOv8在letterbox、坐标还原等方面的差异

#### 相关文档
- [YOLOv8_ONNX_迁移方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/YOLOv8_ONNX_迁移方案.md) - 初步迁移方案
- [YOLOv8_TensorRT_迁移方案_完整版.md](file:///home/seurobot2/Downloads/7_26_V0/docs/YOLOv8_TensorRT_迁移方案_完整版.md) - 完整迁移方案（包含详细代码）
- [TX2_视觉系统迁移至OrinNX_YOLOv8_TensorRT10实施方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/TX2_视觉系统迁移至OrinNX_YOLOv8_TensorRT10实施方案.md) - TRT10专项实施方案

#### 状态说明
✅ **方案设计完成** - 完整的技术方案已产出，待实施

---

### 视觉图像链路一致性核查

#### 工作描述
完成对Vision链路中“debuger发送图像”与“TensorRT推理输入图像”的一致性核查，明确两者关系与尺寸匹配条件：

1. **同源性确认**：两条链路共同来源于 `dev_undis_`
2. **差异点确认**：推理前存在额外 resize 与 BGR->RGB float 转换
3. **尺寸匹配判定**：推理尺寸由 engine 输入shape决定，不由 image.width/height 单独决定

#### 相关文档
- [2026-03-24_视觉debug图与TensorRT输入一致性核查.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-24_视觉debug图与TensorRT输入一致性核查.md) - 图像链路一致性与640x480匹配性结论

#### 状态说明
✅ **已完成** - 已形成代码级证据与结论

---

### 视觉推理letterbox最小改造

#### 工作描述
完成视觉推理几何链路的最小可用对齐改造，重点将推理前输入从直接拉伸改为letterbox，并在后处理中按对称公式做坐标反算与边界clamp：

1. **预处理对齐**：新增 CUDA letterbox resize，返回 scale/pad 元信息
2. **后处理对齐**：TRT解码阶段改为 `(coord - pad) / scale` 反算
3. **边界稳健性**：输出框统一 clamp 到原图范围，避免越界坐标传播

#### 相关文档
- [2026-03-24_视觉推理letterbox对齐最小改造实施记录.md](file:///home/ydf/Robocup/7_26_V0/docs/2026-03-24_视觉推理letterbox对齐最小改造实施记录.md) - 本轮代码改造与验证记录

#### 状态说明
✅ **已完成** - 最小可用改造已落地，待目标设备完整回归

---

### 视觉系统与ZED官方实现对标及去畸变集成

#### 工作描述
基于ZED SDK官方目标检测示例，对标其完整数据流，补完前期改造中缺失的**去畸变处理**环节：

1. **官方对标分析**：调研ZED SDK示例中的完整链路（GPU buffer → 自动去畸变 → letterbox → 推理 → 坐标反算）
2. **P1去畸变集成**：
   - 新增CUDA kernel `undistort_direct_kernal`，支持直接根据畸变参数计算（不依赖预计算映射）
   - 修改Vision ZED路线从跳过去畸变改为启用 `cudaUndistort`
   - 支持三路摄像头分离处理（ZED直接、MV预计算、V4L2无处理）
3. **P2坐标对齐验证**：确认坐标反算逻辑与官方完全一致，无需改动
4. **配置参数化**：新增vision配置块（去畸变开关、padding值、模型类型）

#### 技术亮点
- ✅ 直接Kernel方案：规避了ZED动态参数下预计算映射的复杂性
- ✅ 低侵入改造：仅改4个文件，Vision层无需新增成员
- ✅ 三路隔离：MV/ZED/V4L2图像处理逻辑完全独立，易于扩展与维护

#### 相关文档
- [2026-03-24_视觉系统与ZED官方实现对标及去畸变集成记录.md](2026-03-24_视觉系统与ZED官方实现对标及去畸变集成记录.md) - 完整技术实施与决策记录

#### 状态说明
✅ **已完成** - 代码改造完成，编译通过（P1+P2）| 🔵 待验证（运行时效果）

---

## 未完成工作

### TensorRT 10完整实施

#### 工作描述
基于设计方案，完成从Darknet到TensorRT 10的完整代码实施：

1. **TRTDetector类实现**：使用TensorRT 10 name-based API实现推理封装
2. **预处理CUDA kernel**：实现letterbox预处理（BGR→RGB、归一化、HWC→CHW）
3. **Vision模块集成**：替换vision.hpp和vision.cpp中的Darknet代码
4. **CMake构建系统**：清理Darknet依赖，配置TensorRT链接
5. **配置文件更新**：修改config.conf，指向新的engine文件

#### 待解决问题
- **并发竞态**：camera_src_可见早于GPU缓冲分配的风险需修复
- **流同步**：预处理流与推理流可能不同步，需统一流策略
- **API兼容性**：需确保完全使用TensorRT 10 name-based API，避免使用deprecated的enqueueV2

#### 相关文档
- [YOLOv8_TensorRT_迁移方案_完整版.md](file:///home/seurobot2/Downloads/7_26_V0/docs/YOLOv8_TensorRT_迁移方案_完整版.md) - 完整实施方案
- [TX2_视觉系统迁移至OrinNX_YOLOv8_TensorRT10实施方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/TX2_视觉系统迁移至OrinNX_YOLOv8_TensorRT10实施方案.md) - TRT10实施规范

#### 状态说明
🔄 **待实施** - 方案已完整设计，代码实施待进行

---

### 深度信息融合

#### 工作描述
在ZED-Mini单目稳定基础上，引入深度信息提升测距与定位鲁棒性：

1. **深度数据获取**：启用ZED深度模式，获取深度图
2. **ROI深度融合**：对检测框区域执行深度中值/分位数融合，抑制噪声和空洞
3. **距离估计升级**：用深度测距替代单目几何反投影
4. **异常回退**：深度无效、空洞、强反光时回退到单目几何估计

#### 待解决问题
- **深度质量优化**：需对检测框区域执行深度去噪（中值/双边）与置信度过滤
- **有效深度窗口**：需为球和球门分别设置有效深度窗口，避免极近/极远误值
- **算力开销控制**：深度计算算力开销较高，需降分辨率或限制深度范围

#### 相关文档
- [机器人视觉系统摄像头技术文档与ZED-Mini替换方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/机器人视觉系统摄像头技术文档与ZED-Mini替换方案.md) - 深度融合方案设计

#### 状态说明
🔄 **待实施** - 单目稳定后进行

---

### 性能优化与稳定性增强

#### 工作描述
在功能完成后，进行性能优化和稳定性增强：

1. **内存与拷贝优化**：尽量使用GPU侧图像句柄，减少Host<->Device复制
2. **线程与队列优化**：摄像头采集线程与视觉处理线程解耦，使用无锁队列或双缓冲
3. **推理链路优化**：结合TensorRT batch=1低延迟配置，在ZED输出分辨率与网络输入尺寸之间做一次性缩放
4. **运行稳定性**：增加grab超时、设备断连重试、帧时间戳监控

#### 待解决问题
- **GPU零拷贝验证**：需实测零拷贝是否真正优于显式D2H
- **掉帧监测**：需增加掉帧监测与自动重连机制
- **参数标定**：需完成多光照和高动态场景参数标定

#### 相关文档
- [机器人视觉系统摄像头技术文档与ZED-Mini替换方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/机器人视觉系统摄像头技术文档与ZED-Mini替换方案.md) - 性能优化建议

#### 状态说明
🔄 **待实施** - 功能稳定后进行

---

## 技术文档索引

### 系统架构与设计
- [机器人视觉系统摄像头技术文档与ZED-Mini替换方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/机器人视觉系统摄像头技术文档与ZED-Mini替换方案.md) - 完整的摄像头技术文档和ZED-Mini替换方案，包含架构分析、数据流说明、API接口、迁移路线图

### 迁移方案
- [YOLOv8_ONNX_迁移方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/YOLOv8_ONNX_迁移方案.md) - YOLOv3到YOLOv8 ONNX的初步迁移方案
- [YOLOv8_TensorRT_迁移方案_完整版.md](file:///home/seurobot2/Downloads/7_26_V0/docs/YOLOv8_TensorRT_迁移方案_完整版.md) - YOLOv8 TensorRT完整迁移方案，包含详细代码实现
- [TX2_视觉系统迁移至OrinNX_YOLOv8_TensorRT10实施方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/TX2_视觉系统迁移至OrinNX_YOLOv8_TensorRT10实施方案.md) - 针对TensorRT 10的专项实施方案，强调FP16 Only和彻底去Darknet

### 问题诊断与修复
- [2026-03-23_Qt有画面但无检测框_原因解析.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_Qt有画面但无检测框_原因解析.md) - Qt有画面但无检测框问题的原因分析和验证步骤
- [2026-03-23_x86_debuger_Qt黑屏原因分析与改进方案.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_x86_debuger_Qt黑屏原因分析与改进方案.md) - Qt黑屏问题的完整分析和改进方案
- [2026-03-23_阶段A最小修复_ZED跳过去畸变实施记录.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_阶段A最小修复_ZED跳过去畸变实施记录.md) - ZED跳过去畸变的代码实施记录
- [2026-03-23_ZED_Streaming与Qt调试方案完善日志.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_ZED_Streaming与Qt调试方案完善日志.md) - ZED Streaming与Qt调试方案的完善
- [2026-03-24_视觉debug图与TensorRT输入一致性核查.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-24_视觉debug图与TensorRT输入一致性核查.md) - debug图像与推理图像关系及640x480匹配性核查
- [2026-03-24_视觉推理letterbox对齐最小改造实施记录.md](file:///home/ydf/Robocup/7_26_V0/docs/2026-03-24_视觉推理letterbox对齐最小改造实施记录.md) - 视觉推理链路最小改造实施记录（letterbox+反算+clamp）

### 工具升级
- [2026-03-23_auto_marker_trt10_upgrade_log.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_auto_marker_trt10_upgrade_log.md) - auto_marker工具升级到TensorRT 10的变更记录

### 接入指南
- [ZED-Mini单目接入快速指南.md](file:///home/seurobot2/Downloads/7_26_V0/docs/ZED-Mini单目接入快速指南.md) - ZED-Mini单目接入的快速指南
- [2026-03-23_ZED-Mini单目接入代码改造日志.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_ZED-Mini单目接入代码改造日志.md) - ZED-Mini单目接入的代码改造日志
- [2026-03-23_ZED-Mini单目接入指南_工作日志.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_ZED-Mini单目接入指南_工作日志.md) - ZED-Mini单目接入指南的工作日志

### 逻辑检查与修正
- [2026-03-23_视觉检测逻辑与类别ID一致性检查.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_视觉检测逻辑与类别ID一致性检查.md) - 视觉检测逻辑与类别ID的一致性检查
- [2026-03-23_auto_marker类别映射修正与使用说明.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_auto_marker类别映射修正与使用说明.md) - auto_marker类别映射的修正与使用说明
- [2026-03-23_auto_marker与Qt无框关联分析及代码补充.md](file:///home/seurobot2/Downloads/7_26_V0/docs/2026-03-23_auto_marker与Qt无框关联分析及代码补充.md) - auto_marker与Qt无框问题的关联分析

---

## 使用指南

### 如何使用本索引

1. **快速定位工作内容**：根据"已完成工作"和"未完成工作"两大模块，快速了解项目进展
2. **查找相关文档**：每个工作项下都列出了相关文档的链接，点击即可跳转到具体文档
3. **了解工作状态**：通过状态说明（✅已完成、🔄待实施）了解工作进展
4. **继续特定工作**：根据未完成工作项的描述，找到对应的实施方案文档，继续进行开发

### 后续Agent工作建议

1. **优先处理未完成工作**：按照未完成工作模块的顺序，优先实施TensorRT 10完整实施
2. **参考已有方案**：在实施过程中，参考相关文档中的详细方案和代码示例
3. **记录工作日志**：完成工作后，按照现有日志的格式，记录新的工作日志
4. **更新本索引**：当有新的工作完成时，及时更新本索引文档

---

**文档维护说明**：
- 本文档应随着新工作日志的产生而持续更新
- 每完成一项工作，应及时将相关文档添加到"已完成工作"模块
- 每开始一项新工作，应在"未完成工作"模块中添加相应条目
- 保持文档结构的清晰性和逻辑性，便于后续Agent快速定位和查询
