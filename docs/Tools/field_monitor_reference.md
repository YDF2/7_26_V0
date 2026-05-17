# Field Monitor 参考文档

## 1. 文件结构

```
src/tools/debuger/field_monitor/
├── CMakeLists.txt
├── field_monitor.hpp
└── field_monitor.cpp

src/lib/tcp.hpp                          ← +LOCALIZATION_DATA=13
src/controller/player/vision/vision.cpp   ← 发送定位数据
src/tools/debuger/CMakeLists.txt          ← +add_subdirectory + target_link
src/tools/debuger/debuger.hpp             ← +procBtnFM()
src/tools/debuger/debuger.cpp             ← +按钮 + include + handler
```

## 2. 数据流

```
┌─── Controller (机器人) ───┐          ┌─── Debuger (PC) ────┐
│                            │          │                       │
│  Vision::run()             │  TCP     │  FieldMonitor::       │
│    ↓                       │  ────→   │    data_handler()     │
│  WM->self() → robot_x,y    │          │    ↓                  │
│  WM->ball() → ball_x,y     │  33B/帧  │  emit updated()      │
│                            │          │    ↓                  │
│  SERVER->write(cmd)        │          │  paintEvent()         │
└────────────────────────────┘          └───────────────────────┘
```

**发送条件**：`OPTS->use_debug()` 且 `OPTS->use_robot()` 都为 true。
x86 调试模式（`-r false`）不发送，因为没有 WM 定位数据。

## 3. 协议格式

每条消息 = `remote_data_type`(4B) + payload(29B)

```
偏移  类型    字段
──────────────────────
 0    enum    LOCALIZATION_DATA (=13)
 4    float   robot_x       全局坐标 (cm)
 8    float   robot_y
12    float   robot_dir     朝向 (deg)
16    float   ball_x        球全局坐标 (cm)
20    float   ball_y
24    float   ball_self_x   球在机器人系下坐标 (cm) — 当前未显示
28    float   ball_self_y
32    bool    ball_can_see
```

## 4. 场地参数

硬编码在 `FieldMonitor()` 构造函数中，不读取 `field.conf`：

```cpp
field_length_           = 600;   // 场长 (cm)
field_width_            = 400;   // 场宽 (cm)
goal_width_             = 150;   // 球门宽
goal_depth_             = 50;    // 球门深
penalty_area_length_    = 100;   // 大禁区纵深
penalty_area_width_     = 250;   // 大禁区宽
goal_area_length_       = 50;    // 小禁区纵深
goal_area_width_        = 150;   // 小禁区宽
penalty_mark_distance_  = 100;   // 罚球点到门线
center_circle_diameter_ = 120;   // 中圈直径
border_strip_width_     = 70;    // 边界留白
```

**修改方式**：直接改构造函数里的值，或改成从 `field.conf` 读取。

## 5. 坐标系

```
Field 坐标系 (cm):              Qt 窗口坐标系 (px):
  x → 长度方向 (0=中线)           x→ (不变)
  y → 宽度方向 (0=中线)           y↓ (翻转，fy = -field_y)

转换函数 (field_monitor.cpp:165-166):
  fx(field_x) = field_x
  fy(field_y) = -field_y

球门位置:
  左门: x = -300, y = 0      (field)
  右门: x = +300, y = 0      (field)
```

## 6. 绘制函数

### drawField(painter)
按顺序绘制：边线 → 中线 → 中圈 → 开球点 → 左大禁区 → 左小禁区 → 右大禁区 → 右小禁区 → 左球门 → 右球门 → 左罚球点 → 右罚球点

**想加新线/删线**：直接在这函数里增删 `drawLine`/`drawRect`/`drawEllipse`。

### drawBall(painter)
- visible → 橙色实心圆
- lost → 灰色虚线空心圆
- 球半径：`ball_r_ = 5`（固定像素，不随缩放变化；想让它缩放改成 `ball_r_ * scale_`）

### drawRobot(painter)
- 白色方块 + 向前方向的线
- 方块半边长：`robot_sq_ = 6`
- 方向线长：`robot_sq_ + 8`

## 7. 窗口行为

```cpp
int base_w = field_length_ + 2 * border_strip_width_;   // 740
int base_h = field_width_  + 2 * border_strip_width_;   // 540
resize(static_cast<int>(base_w * 1.2), ...);             // 初始 1.2x
setMinimumSize(500, 400);                                // 最小尺寸
```

缩放逻辑在 `paintEvent()`：
```cpp
float sx = width()  / base_w;
float sy = height() / base_h;
scale_ = std::min(sx, sy);
p.scale(scale_, scale_);    // QPainter 全局缩放，所有元素自动等比
```

**想改成固定大小**：加 `setFixedSize(...)`，去掉 `scale_` 计算。
**想改缩放比例**：改初始 `resize` 的 `1.2` 倍率。
**想让球/机器人也随窗口缩放**：目前球半径和方块大小在 `p.scale()` 之后绘制，会被 scaling 影响，所以实际是随窗口缩放的。

## 8. 颜色 / 样式

```cpp
setStyleSheet("background:rgb(0,110,0)");  // 场地底色（深绿）

// 球可见
p.setPen(QPen(QColor(255, 140, 0), 2));    // 橙色边框
p.setBrush(QBrush(QColor(255, 140, 0)));   // 橙色填充

// 球丢失
p.setPen(QPen(Qt::gray, 1, Qt::DotLine));  // 灰色虚线
p.setBrush(Qt::NoBrush);                   // 空心

// 机器人 + 场地线
QPen(Qt::white, 2)                         // 白色，线宽 2px
```

**改颜色**：直接改 `Qt::white` / `QColor(r,g,b)` / `Qt::gray` 等。

## 9. TCP 连接

- 连接地址和端口从 `config.conf` 读取（与 ImageMonitor 相同）
- 绿色圆点 = 已连接，红色 = 断连
- 连接后 1 秒自动注册 `REMOTE_DATA`
- 使用 `data_handler` 回调，在 TCP 线程中调用，通过 signal `localizationUpdated` 触发主线程 `update()`

**注意**：`data_handler` 在 TCP 线程，不能直接操作 GUI。更新通过 emit signal + `Qt::AutoConnection`（跨线程自动排队到主线程）。

## 10. 常见修改场景

| 需求 | 改哪里 |
|------|--------|
| 改场地尺寸 | 构造函数里的 `field_length_` 等 |
| 加新线条/标记 | `drawField()` |
| 改球的颜色/大小 | `drawBall()` 里的 `QPen`/`ball_r_` |
| 改机器人的形状/颜色 | `drawRobot()` 里的 `QPen`/`QBrush`/方块改三角 |
| 改窗口初始大小 | `resize(base_w * 1.2, ...)` 里的倍率 |
| 改成不能缩放 | 加 `setFixedSize(base_w, base_h)` |
| 显示球相对机器人坐标 | `drawBall` 用 `ball_self_x/y` 而非 `ball_x/y` |
| 加轨迹轨迹线 | 在 `data_handler` 存历史坐标，`paintEvent` 画折线 |
| 从 `field.conf` 读取参数 | `parser::parse(CONF->field_file(), ...)` 替代硬编码 |

## 11. 编译

Field Monitor 作为 debuger 的子目录编译。`field_monitor/CMakeLists.txt`：

```cmake
add_library(field_monitor field_monitor.cpp)
target_link_libraries(field_monitor tcp_client Qt5::Widgets opencv_core pthread)
```

上层 `debuger/CMakeLists.txt`：
```cmake
add_subdirectory(field_monitor)      # 第 26 行
# ...
target_link_libraries(${TARGET_NAME} ... field_monitor ...)
```

修改后重新 cmake + make 即可。
