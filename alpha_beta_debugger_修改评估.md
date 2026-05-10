# alpha/beta 在 Debugger 显示 — 修改评估

## 目标

在 ImageMonitor 窗口下，只要视野中有球，就在状态栏显示 alpha 和 beta 的实时值。

## 现状分析

### alpha/beta 含义

```cpp
// vision.cpp:299-300
float alpha = (ball_pix.x() - params_.cx) / (float)w_;
float beta  = (ball_pix.y() - params_.cy) / (float)h_;
```

| 变量 | 含义 | 方向约定 |
|---|---|---|
| alpha | 球心相对图像中心的水平偏移(归一化) | 正值 = 球偏右 (左负右正) |
| beta | 球心相对图像中心的垂直偏移(归一化) | 正值 = 球偏下 (上小下大) |

### alpha/beta 被计算的条件 (vision.cpp:206-301)

```cpp
if (OPTS->use_robot())          // 条件1: robot 模式
{
    camera_matrix = camera_matrix_;    // 条件2: camera_matrix_ 已构建(需IMU)
    head_yaw = head_yaw_;
    head_pitch = head_pitch_;

    // ... odometry / WM update ...

    if (!ball_dets_.empty())    // 条件3: 检测到球
    {
        float alpha = ...;
        float beta = ...;
        WM->set_ball_pos(..., alpha, beta, true);
    }
}
```

**三个条件缺一不可**：`-r true`(默认) + IMU 串口通信正常 + 视野中有球。

### 当前数据流

```
┌─────────────────── Robot(真机) ───────────────────┐
│                                                    │
│  Vision::run()  计算 alpha/beta                    │
│       │                                            │
│       ├──→ WM->set_ball_pos()  (存到 ball_block)    │
│       │                                            │
│       └──→ send_image(bgr) ──TCP──→ ImageMonitor   │
│            (只传输 JPEG 图像，不含 alpha/beta)        │
│                                                    │
└────────────────────────────────────────────────────┘
```

**问题**：alpha/beta 只存到了本地 WM，从未发送给 debugger。

### 现有 TCP 消息类型可复用情况

`tcp.hpp` 中已有的相关枚举：

```cpp
enum image_send_type {
    IMAGE_SEND_ORIGIN = 0,
    IMAGE_SEND_RESULT = 1,
    IMAGE_SEND_RECT   = 2,
    IMAGE_SEND_POINT  = 3,
    IMAGE_SEND_DIS    = 4   // 当前收到 x,y 但未在 UI 显示，可改造
};
```

`IMAGE_SEND_DIS` 目前被 Vision::get_point_dis() 用于回传点击位置的测距结果(x,y)，但 debugger 端收到后只读了值、没有显示。可以复用这个通道，也可以新增类型。

## 修改方案

### 总体思路

1. Vision 端：alpha/beta 变为**成员变量**持久保存，每帧球可见时更新
2. 发送时机：与 send_image 同步，球可见时把 alpha/beta 通过 REMOTE_DATA 发到 debugger
3. Debugger 端：接收后在 statusBar 显示

### 文件改动清单 (5 个文件)

#### 1. `src/lib/tcp.hpp` — 新增消息类型

```cpp
// 在 image_send_type 枚举中新增:
IMAGE_SEND_BALL = 5   // [float alpha][float beta][bool can_see]
```

#### 2. `src/controller/player/vision/vision.hpp` — 新增成员变量

```cpp
float last_alpha_;
float last_beta_;
bool ball_visible_;
```

#### 3. `src/controller/player/vision/vision.cpp` — 核心改动

**改动 A（line ~293-301）**：将 alpha/beta 计算提为成员变量

```cpp
// 原代码在 if (OPTS->use_robot()) 内部
// 改为: 只要 ball_dets_ 非空，就计算 alpha/beta 并保存到成员变量

if (!ball_dets_.empty())
{
    Vector2i ball_pix(ball_dets_[0].x + ball_dets_[0].w / 2, ball_dets_[0].y + ball_dets_[0].h);
    last_alpha_ = (ball_pix.x() - params_.cx) / (float)w_;
    last_beta_  = (ball_pix.y() - params_.cy) / (float)h_;
    ball_visible_ = true;
}
else
{
    ball_visible_ = false;
}

// 下面的 odometry/WM 逻辑保持在 if (OPTS->use_robot()) 中不动
if (OPTS->use_robot())
{
    self_block p = WM->self();
    if (!ball_dets_.empty())
    {
        // odometry + WM->set_ball_pos(..., last_alpha_, last_beta_, true);
    }
}
```

**改动 B（line ~380-438）**：发送 alpha/beta 到 debugger

```cpp
if (OPTS->use_debug())
{
    // 在发送图像之前/之后，发送 alpha/beta
    if (ball_visible_)
    {
        tcp_command cmd;
        cmd.type = REMOTE_DATA;
        cmd.size = 2 * enum_size + 2 * float_size + bool_size;
        remote_data_type t1 = IMAGE_SEND_TYPE;
        image_send_type t2 = IMAGE_SEND_BALL;
        cmd.data.clear();
        cmd.data.append((char *)&t1, enum_size);
        cmd.data.append((char *)&t2, enum_size);
        cmd.data.append((char *)&last_alpha_, float_size);
        cmd.data.append((char *)&last_beta_, float_size);
        cmd.data.append((char *)&ball_visible_, bool_size);
        SERVER->write(cmd);
    }
    // ... 原有 send_image 代码不变 ...
}
```

#### 4. `src/tools/debuger/image_monitor/image_monitor.hpp` — 新增 UI 元素

```cpp
QLabel *alphaLab, *betaLab;
```

#### 5. `src/tools/debuger/image_monitor/image_monitor.cpp` — 接收并显示

**改动 A（构造函数）**：新增两个 label 加到 statusBar

```cpp
alphaLab = new QLabel("alpha: --");
betaLab  = new QLabel("beta: --");
alphaLab->setFixedWidth(130);
betaLab->setFixedWidth(130);
statusBar()->addWidget(alphaLab);
statusBar()->addWidget(betaLab);
```

**改动 B（data_handler）**：处理 IMAGE_SEND_BALL 消息

```cpp
else if (cmd.type == REMOTE_DATA)
{
    remote_data_type t1;
    memcpy(&t1, cmd.data.c_str(), enum_size);
    if (t1 == IMAGE_SEND_TYPE)
    {
        image_send_type t2;
        memcpy(&t2, cmd.data.c_str() + enum_size, enum_size);
        if (t2 == IMAGE_SEND_BALL)
        {
            float alpha, beta;
            bool can_see;
            memcpy(&alpha, cmd.data.c_str() + 2*enum_size, float_size);
            memcpy(&beta,  cmd.data.c_str() + 2*enum_size + float_size, float_size);
            memcpy(&can_see, cmd.data.c_str() + 2*enum_size + 2*float_size, bool_size);
            if (can_see)
            {
                alphaLab->setText(QString("alpha: %1").arg(alpha, 0, 'f', 4));
                betaLab->setText( QString("beta: %1").arg(beta,  0, 'f', 4));
            }
            else
            {
                alphaLab->setText("alpha: --");
                betaLab->setText("beta: --");
            }
        }
        // ... 原有 IMAGE_SEND_DIS 等分支保持不变 ...
    }
}
```

### 改动量估算

| 文件 | 新增行数 | 修改行数 | 风险 |
|---|---|---|---|
| tcp.hpp | +1 | 0 | 低 |
| vision.hpp | +3 | 0 | 低 |
| vision.cpp | +20 | ~5 (移动代码块) | 中 |
| image_monitor.hpp | +1 | 0 | 低 |
| image_monitor.cpp | +20 | 0 | 低 |

### 不改变的行为

- 原有 use_robot() 下的 odometry/WM 更新逻辑完全不变
- x86 纯调试模式（`-r false`）下 alpha/beta 原本不计算，**现在会计算并发送到 debugger**（这是预期的新功能）
- 真机模式下 alpha/beta 计算语义不变，只是额外发送了一份到 debugger
- IMAGE_SEND_DIS 原有逻辑不动（保留兼容）
- 图像发送频率不变

### 潜在风险

1. **CPU 周期开销**：每 50ms 多发送一个 REMOTE_DATA 小包，开销可忽略（约 30 字节）
2. **TCP 粘包**：REMOTE_DATA 和 IMG_DATA 分开发送，两者到达 debugger 的时序可能不同。由于 debugger 端用 `data_handler` 回调分别处理，各自独立，不会乱
3. **编译依赖**：tcp.hpp 的枚举同时被 controller 和 tools/debuger 引用，新增枚举值后两边都需重新编译

---

## 实施步骤建议

1. 修改 `tcp.hpp` 新增枚举
2. 修改 `vision.hpp` 新增成员变量
3. 修改 `vision.cpp`：提取 alpha/beta 计算 + 新增发送逻辑
4. 修改 `image_monitor.hpp/.cpp`：新增 UI + 接收逻辑
5. 编译验证（x86_64）
