# Field Monitor 设计文档

## 1. 目标

在 debuger 中新增一个 **Field Monitor** 窗口，实时显示：
- 足球场俯视图（含门柱、禁区线、中圈）
- 机器人自身位置 + 朝向（三角形箭头）
- 球的位置（圆点，颜色区分可见/丢失）
- 数据通过 TCP 连接，与 ImageMonitor 共用同一 TCP 链路

## 2. 现有相关代码

| 组件 | 文件 | 说明 |
|------|------|------|
| TeamMonitor | `team_monitor/` | 已有！UDP 广播显示全场多机器人 + 球 |
| soccer map | `localization/soccermap.h` | 场地尺寸（cm），门柱坐标 |
| ball_block | `lib/model.hpp:75` | `global(x,y)`, `can_see` |
| self_block | `lib/model.hpp:84` | `global(x,y)`, `dir` |
| field.conf | `src/data/model/field.conf` | 场地参数：900×600cm |

**与 TeamMonitor 的区别：**
- TeamMonitor 走 UDP 广播，看全场所有机器人
- FieldMonitor 走 TCP，看**当前连接的单台机器人**的高频定位数据，与 ImageMonitor 同链路

## 3. 数据路径

```
Robot (Controller)                          Debuger (PC)
─────────────────                          ────────────
Vision::run()
  │
  ├─ ball_block_.global                    tcp_command
  ├─ self_block_.global       ──TCP──→    data_handler()
  └─ self_block_.dir                        │
       │                                    ├─ LOCALIZATION_DATA
       ▼                                    │   → update robot_pos
  SERVER->write(cmd)                        │   → update ball_pos
                                            │   → repaint()
```

## 4. TCP 协议新增

### 4.1 `remote_data_type` 新增枚举（`src/lib/tcp.hpp`）

```cpp
enum remote_data_type
{
    // ... 现有 ...
    IMAGE_SEND_TYPE = 12,
    LOCALIZATION_DATA = 13,   // 新增
};
```

### 4.2 数据结构

发送端每条消息 = 1 个 `remote_data_type`(4B) + 1 个 `localization_payload`(28B)

```cpp
struct localization_payload
{
    float robot_x;      // 机器人全局 x (cm)
    float robot_y;      // 机器人全局 y (cm)
    float robot_dir;    // 机器人朝向 (deg)
    float ball_x;       // 球全局 x (cm)
    float ball_y;       // 球全局 y (cm)
    float ball_self_x;  // 球在机器人系下 x (cm)
    float ball_self_y;  // 球在机器人系下 y (cm)
    bool  ball_can_see; // 球是否可见
};
// payload_size = 7 * sizeof(float) + sizeof(bool) = 28 + 1 = 29
// 加上 remote_data_type 头部 4B = 33B
```

### 4.3 发送时机

在 `Vision::run()` 中，`if (OPTS->use_debug())` 块内，与 IMAGE_SEND_BALL 一起发送。

### 4.4 发送代码

```cpp
if (OPTS->use_debug())
{
    // 发送定位数据
    if (OPTS->use_robot())
    {
        self_block self = WM->self();
        ball_block ball = WM->ball();

        tcp_command cmd;
        cmd.type = REMOTE_DATA;
        remote_data_type t1 = LOCALIZATION_DATA;
        cmd.size = enum_size + 7 * float_size + bool_size;
        cmd.data.clear();
        cmd.data.append((char *)&t1, enum_size);
        cmd.data.append((char *)&self.global.x(), float_size);
        cmd.data.append((char *)&self.global.y(), float_size);
        cmd.data.append((char *)&self.dir, float_size);
        cmd.data.append((char *)&ball.global.x(), float_size);
        cmd.data.append((char *)&ball.global.y(), float_size);
        cmd.data.append((char *)&ball.self.x(), float_size);
        cmd.data.append((char *)&ball.self.y(), float_size);
        cmd.data.append((char *)&ball.can_see, bool_size);
        SERVER->write(cmd);
    }

    // ... 原有 IMAGE_SEND_BALL 和图像发送不变 ...
}
```

## 5. FieldMonitor 窗口设计

### 5.1 显示内容

```
┌────────────────────────────────────────────┐
│  [Field Monitor]  192.168.x.x:xxxx         │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────── 绿色场地 ────────────┐      │
│  │  ┌── 球门(上) ──┐              │      │
│  │  │              │              │      │
│  │  └──────────────┘              │      │
│  │  禁区线                         │      │
│  │                                  │      │
│  │          ○ (球，橙色)            │      │
│  │              ▲ (机器人，蓝色三角形)│      │
│  │              │                   │      │
│  │        中圈  │                   │      │
│  │                                  │      │
│  │  禁区线                         │      │
│  │  ┌── 球门(下) ──┐              │      │
│  │  │              │              │      │
│  │  └──────────────┘              │      │
│  └──────────────────────────────────┘      │
│                                            │
│  robot: x= 120.0  y= -80.0  dir= 45.0°    │
│  ball:  x= 320.0  y=  50.0  visible       │
├────────────────────────────────────────────┤
│  robot: 120.0, -80.0, 45.0°                │
└────────────────────────────────────────────┘
```

### 5.2 状态栏

```
robot: x=xxx.x  y=xxx.x  dir=xxx.x°  |  ball: x=xxx.x  y=xxx.x  ○/✕
```

### 5.3 场地参数

标准 RoboCup Kidsize 决赛/正式联赛场地：

| 参数 | 值 (cm) | 说明 |
|------|---------|------|
| field_length | 600 | 场地长 |
| field_width | 400 | 场地宽 |
| goal_width | 150 | 球门宽 |
| goal_depth | 50 | 球门深 |
| goal_area_length | 100 | 禁区纵深 |
| goal_area_width | 250 | 禁区宽 |
| center_circle_diameter | 120 | 中圈直径 |
| penalty_mark_distance | 100 | 罚球点到门线 |
| border_strip_width_min | 70 | 边界留白 |

**注意：当前 `src/data/model/field.conf` 是 900×600（旧），需要同步修改为上述值。**

### 5.4 需绘制的场地线条

```
         ┌────── 球门(上) ──────┐
         │    ┌──────────┐     │
         │    │   禁区   │     │
         ├────┤  ┌──┐   ├─────┤
         │    │  │罚│   │     │
         │    │  │球│   │     │
    ─────┼────┼──┤点├───┼─────┼───── 中线
         │    │  └──┘   │     │
         │    │  中圈   │     │
         │    └──────────┘     │
         │     禁区(下)        │
         └────── 球门(下) ──────┘
```

绘制清单：
1. **边线** — 600×400 矩形
2. **中线** — 横向贯穿
3. **中圈** — 直径 120cm 圆，圆心在场中心
4. **禁区（上方）** — 100×250 矩形，贴上方边线
5. **禁区（下方）** — 100×250 矩形，贴下方边线
6. **球门（上方）** — 50×150 矩形，贴上方边线外侧
7. **球门（下方）** — 50×150 矩形，贴下方边线外侧
8. **罚球点（上）** — 距上方球门线 100cm
9. **罚球点（下）** — 距下方球门线 100cm
10. **开球点** — 场地中心

### 5.5 颜色约定

| 元素 | 颜色 | 说明 |
|------|------|------|
| 场地背景 | 深绿色 | RGB(0, 128, 0) |
| 场地线 | 白色 | 宽 2px |
| 机器人 | 白色 | 方块（边长自适应缩放） |
| 机器人朝向 | 白色直线 | 从方块中心向前延伸 |
| 球（可见） | 橙色 | 实心圆 |
| 球（丢失） | 灰色 | 虚线空心圆 |
| 球门框 | 白色 | 宽 2px |
| 禁区线 | 白色 | 宽 2px |

### 5.4 坐标系转换

```
场地坐标 (x_cm, y_cm):           Qt 绘图坐标 (px, py):
  x → 场地长度方向 (900cm)          px =  x_cm * scale
  y → 场地宽度方向 (600cm)          py = -y_cm * scale  (翻转)
  原点 = 场地中心                  原点 = widget 中心
```

scale 自适应窗口大小，保持场地比例。

### 5.5 交互

- 鼠标滚轮：缩放
- 鼠标拖拽：平移视野（可选）
- 按 C：清除轨迹（可选）

## 6. 文件改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/lib/tcp.hpp` | 修改 | +`LOCALIZATION_DATA = 13` |
| `src/controller/player/vision/vision.cpp` | 修改 | +localization 数据发送 (~20行) |
| `src/tools/debuger/field_monitor/CMakeLists.txt` | 新增 | 编译配置 |
| `src/tools/debuger/field_monitor/field_monitor.hpp` | 新增 | 类声明 |
| `src/tools/debuger/field_monitor/field_monitor.cpp` | 新增 | 实现 (~200行) |
| `src/tools/debuger/CMakeLists.txt` | 修改 | +field_monitor 子目录 |
| `src/tools/debuger/debuger.hpp` | 修改 | +procBtnFM 声明 |
| `src/tools/debuger/debuger.cpp` | 修改 | +按钮 + 处理 |

## 7. FieldMonitor 类设计

```cpp
class FieldMonitor : public QMainWindow
{
    Q_OBJECT
public:
    FieldMonitor();
    void data_handler(const tcp_command cmd);

signals:
    void closed();
    void localizationUpdated();  // 触发重绘

protected:
    void paintEvent(QPaintEvent *event);
    void closeEvent(QCloseEvent *event);
    void wheelEvent(QWheelEvent *event);

private:
    void drawField(QPainter &p);
    void drawRobot(QPainter &p);
    void drawBall(QPainter &p);

    tcp_client client_;

    // 场地参数（从配置读）
    int field_length_cm_;
    int field_width_cm_;
    int goal_width_cm_;
    int goal_depth_cm_;
    int goal_area_length_cm_;
    int goal_area_width_cm_;
    int penalty_mark_cm_;
    int center_circle_diameter_cm_;

    // 机器人状态
    float robot_x_, robot_y_, robot_dir_;
    bool robot_valid_;

    // 球状态
    float ball_x_, ball_y_;
    bool ball_can_see_;
    bool ball_valid_;

    // 显示参数
    float scale_;
    QLabel *robotLabel, *ballLabel;

    // --- 显示参数 ---
    float robot_square_size_;     // 机器人白色方块大小 (cm)，建议 12cm
    float ball_radius_;           // 球半径 (cm)，标准直径 10cm → 半径 5cm
};
```

## 8. 待用户提供参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| 机器人身体尺寸 | 长 × 宽 (cm)，用于画三角形/矩形 | 20 × 15 cm |
| 球直径 | 用于画球的圆半径 (cm) | 10 cm |
| 球颜色 | 可见/丢失颜色 | 橙/灰 |
| 机器人颜色 | 三角形颜色 | 蓝 |
| 缩放初始值 | 默认 scale | 自适应窗口 |

## 9. paintEvent 绘制流程

```
1. QPainter 初始化，translate 到 widget 中心
2. drawField(painter):
   - 绿色背景矩形 (field_length × field_width)
   - 中线、中圈
   - 上下两个禁区矩形
   - 上下两个球门矩形
   - 罚球点
   - 边线
3. drawBall(painter):
   - translate 到球位置
   - can_see ? 橙色实心圆 : 灰色空心圆
   - scale = ball_display_size / 2
4. drawRobot(painter):
   - translate 到机器人位置
   - rotate(robot_dir)
   - 画蓝色三角形 + 朝向线
5. 状态栏更新
```

## 10. 与现有系统的兼容

- 不影响 ImageMonitor、TeamMonitor 等其他 debuger 窗口
- 不与 FSM / WM 逻辑冲突
- `use_robot()=false` 时不发送定位数据（无数据可发）
- TCP 链路已注册 REMOTE_DATA，FieldMonitor 只需额外注册 LOCALIZATION_DATA
