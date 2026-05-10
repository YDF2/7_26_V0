# Ball Alpha/Beta EKF 设计文档 v3

## 1. 背景

### 1.1 当前数据流

```
YOLO 检测 → ball_dets_
              │
              ├─ 非空 → 算 raw alpha/beta → last_alpha_/last_beta_
              │         ball_visible_ = true
              │         WM->set_ball_pos(..., true)   ← 含 odometry
              │         cant_see_ball_count_ = 0
              │
              └─ 空   → ball_visible_ = false
                        cant_see_ball_count_++
                        500ms 后 WM->set_ball_pos(0,..., false)
```

alpha/beta 的消费者：

| 消费者 | 路径 | 用途 |
|--------|------|------|
| FSM (KICK_BALL) | `WM->ball().alpha/beta` | 踢球前精细平移调整 |
| FSM (DRIBBLE) | `WM->ball().alpha/beta` | 带球时左右控球 |
| FSM (全状态) | `WM->ball().can_see` | 是否看到球 → 决定状态转移 |
| Debugger | TCP `IMAGE_SEND_BALL` | 状态栏显示 |

### 1.2 要解决的问题

| 问题 | 触发场景 | 后果 |
|------|---------|------|
| 踏步自运动噪声 | KICK_BALL 小步调整时身体 ±1~2° 晃动，球像抖动 5-10px | alpha/beta 剧烈跳动，精细阈值判断不准 |
| 短暂丢帧 | 脚遮挡 / 球滚到画面边缘 / YOLO 漏检 1-3 帧 | can_see 立刻变 false，FSM 误切回 SEARCH_BALL |

### 1.3 不改的东西

- `ball_dets_` 的语义：永远是 YOLO 原始检测结果
- `WM->set_ball_pos()` 的接口和全局坐标计算
- FSM 各状态的转移条件和阈值
- Debugger TCP 协议

## 2. BallEKF 设计

### 2.1 状态向量与模型

```
x = [alpha, beta, valpha, vbeta]^T     (4×1)

F = [ 1  0  dt  0 ]          dt = vision_period（典型 0.05s）
    [ 0  1  0  dt]
    [ 0  0  1   0 ]
    [ 0  0  0   1 ]

H = [ 1  0  0  0 ]          线性观测，退化为标准 KF
    [ 0  1  0  0 ]
```

### 2.2 自适应观测噪声 R

机器人踏步晃动是主要噪声源。用 IMU pitch/roll 帧间变化率度量晃动：

```
pitch_rate = |imu.pitch - prev_imu.pitch| / dt     (deg/s)
roll_rate  = |imu.roll  - prev_imu.roll|  / dt

shake = pitch_rate + roll_rate

sigma_meas^2 = sigma_base^2 + k_shake * shake
R = sigma_meas^2 * I_2×2
```

| 场景 | shake 典型值 | sigma_meas | 行为 |
|------|-------------|-----------|------|
| 静止站立 | ~0 | 0.002 | 紧跟随检测值 |
| 小步微调 (KICK) | ~30 deg/s | 0.0055 | 中等平滑 |
| 大步走 (GOTO) | ~60 deg/s | 0.0077 | 较强平滑 |

### 2.3 过程噪声 Q

基于 CWNA 模型，q_c 固定：

```
Q = G * q_c * G^T
G = [dt^2/2, 0; 0, dt^2/2; dt, 0; 0, dt]
q_c = 0.005
```

### 2.4 参数总表

| 参数 | 值 | 含义 |
|------|-----|------|
| dt | 0.05s | vision_period |
| sigma_base | 0.002 | 静止检测噪声（归一化，约 1.3px） |
| k_shake | 1e-6 | 晃动→R 缩放 |
| q_c | 0.005 | 过程噪声强度 |
| P0_diag | [0.01, 0.01, 0.001, 0.001] | 初始协方差 |
| coast_max | 4 | 预测最大帧数（200ms） |
| gate_thresh | 9.0 | 马氏距离平方门限 |

## 3. BallEKF 状态机

### 3.1 状态定义

```
           reinit()                  predict()
UNINIT ────────────→ TRACKING ────────────────────→ TRACKING
   ▲                     │   │                          │
   │     coast_cnt>4     │   │ update() rejected        │
   │    (keep P,clear x) │   └──────────┐               │
   │                     │              │               │
   │                     │ update() accepted            │
   │                     ▼              ▼               │
   │                  TRACKING      TRACKING            │
   │                     │                              │
   │                     │ ball_dets_空                 │
   │                     ▼                              │
   │                  COASTING ──→ predict() ──→ COASTING
   │                     │                              │
   │                     │ ball_dets_非空 + gate OK     │
   │                     └──────────────────→ TRACKING  │
   │                     │                              │
   │                     │ coast_cnt>4                  │
   └─────────────────────┘                              │
```

### 3.2 各状态行为

| 状态 | 条件 | 动作 |
|------|------|------|
| UNINIT | 从未见过球 / 丢失太久 | 等待首次检测 |
| TRACKING | 球可见，EKF 正常跟踪 | predict + update，coast_cnt=0 |
| COASTING | 球暂时丢失 | predict only，coast_cnt++ |

**进入 COASTING**：`ball_dets_` 为空且当前在 TRACKING 态。
**退出 COASTING**：下一帧 `ball_dets_` 非空 + 马氏门通过 → 回到 TRACKING。
**COASTING→UNINIT**：coast_cnt > 4 → 重置状态、清零 x、保留 P（下次检测重新 reinit）。

### 3.3 马氏距离门限（仅在 COASTING 后重新检测时判断）

```
y = z - H*x^-           新息
S = H*P^-*H^T + R       新息协方差
d² = y^T * S^{-1} * y   马氏距离平方

d² < 9.0  → 接受，正常 update
d² ≥ 9.0  → 拒绝，保持 COASTING（可能是误检）
```

在 TRACKING 态时不走门限，直接 update（连续跟踪中帧间跳跃是自运动造成，R 已自适应处理）。

### 3.4 对外接口

```cpp
struct State { float alpha, beta, valpha, vbeta; };

void  init();
void  predict(float dt);
void  update(float alpha, float beta);   // 在 TRACKING 直接更新；在 COASTING 走门限判断；在 UNINIT 转 reinit
void  set_imu_shake(float pitch_rate, float roll_rate);  // 更新自适应 R

State state() const;
bool  is_tracking() const;   // TRACKING 或 COASTING 返回 true
int   coast_count() const;   // COASTING 帧数，TRACKING 返回 0，UNINIT 返回 -1
```

## 4. 集成方案：vision.cpp 改动

### 4.1 vision.hpp 新增

```cpp
#include "ball_ekf.hpp"

// 新增成员：
BallEKF ball_ekf_;
Imu::imu_data prev_imu_;
float imu_shake_pitch_rate_;
float imu_shake_roll_rate_;
```

### 4.2 Vision::start() 新增

```cpp
ball_ekf_.init();
prev_imu_ = Imu::imu_data{};  // 零初始化
imu_shake_pitch_rate_ = 0;
imu_shake_roll_rate_ = 0;
```

### 4.3 Vision::updata() 改动 — IMU 数据到达时计算晃动率

在 SENSOR_IMU 分支中，新增晃动率计算：

```cpp
if (type == Sensor::SENSOR_IMU)
{
    shared_ptr<Imu> sptr = dynamic_pointer_cast<Imu>(pub);
    Imu::imu_data current = sptr->data();
    
    // 计算晃动率（deg/s），仅当有历史数据时
    if (prev_imu_.timestamp != 0) {
        float dt_imu = (current.timestamp - prev_imu_.timestamp) / 1000.0f;
        if (dt_imu > 0 && dt_imu < 1.0f) {
            imu_shake_pitch_rate_ = fabs(current.pitch - prev_imu_.pitch) / dt_imu;
            imu_shake_roll_rate_  = fabs(current.roll  - prev_imu_.roll)  / dt_imu;
        }
    }
    prev_imu_ = current;
    
    // ... 原有 imu_datas_ 队列逻辑不变 ...
}
```

### 4.4 Vision::run() 核心改动（L286-320 区域）

**替换现有的 alpha/beta 计算和 WM 更新逻辑：**

```cpp
// ===== 替代 L286-320 =====

// 1. 更新 EKF 自适应 R（用最新 IMU 晃动率）
ball_ekf_.set_imu_shake(imu_shake_pitch_rate_, imu_shake_roll_rate_);

// 2. 每帧预测
ball_ekf_.predict(period_ms_ / 1000.0f);

// 3. 有检测 → 送入 EKF
bool raw_detected = false;
float raw_alpha = 0, raw_beta = 0;
if (!ball_dets_.empty())
{
    Vector2i ball_pix(ball_dets_[0].x + ball_dets_[0].w / 2,
                      ball_dets_[0].y + ball_dets_[0].h);
    raw_alpha = (ball_pix.x() - params_.cx) / (float)w_;
    raw_beta  = (ball_pix.y() - params_.cy) / (float)h_;
    raw_detected = true;
    
    ball_ekf_.update(raw_alpha, raw_beta);
}

// 4. 输出：一律用 EKF 滤波值
last_alpha_ = ball_ekf_.state().alpha;
last_beta_  = ball_ekf_.state().beta;

// 5. ball_visible_：EKF 在跟踪（TRACKING 或 COASTING ≤4帧）就认为可见
ball_visible_ = ball_ekf_.is_tracking();

// 6. WM 更新
if (OPTS->use_robot())
{
    self_block p = WM->self();
    
    if (ball_ekf_.is_tracking())
    {
        // EKF 在跟踪中：始终向 WM 报告球可见
        
        if (!ball_dets_.empty())
        {
            // 有真实检测 → 用真实像素做 odometry
            Vector2i ball_pix(ball_dets_[0].x + ball_dets_[0].w / 2,
                              ball_dets_[0].y + ball_dets_[0].h);
            Vector2d odo_res = odometry(ball_pix, camera_matrix);
            Vector2d ball_pos = camera2self(odo_res, head_yaw);
            Vector2d temp_ball = p.global + rotation_mat_2d(-p.dir) * ball_pos;
            WM->set_ball_pos(temp_ball, ball_pos, ball_pix,
                             last_alpha_, last_beta_, true);
        }
        else
        {
            // COASTING：无真实像素，用 EKF 预测值反推像素做 odometry
            float px = last_alpha_ * w_ + params_.cx;
            float py = last_beta_  * h_ + params_.cy;
            Vector2i ball_pix(px, py);
            Vector2d odo_res = odometry(ball_pix, camera_matrix);
            Vector2d ball_pos = camera2self(odo_res, head_yaw);
            Vector2d temp_ball = p.global + rotation_mat_2d(-p.dir) * ball_pos;
            WM->set_ball_pos(temp_ball, ball_pos, ball_pix,
                             last_alpha_, last_beta_, true);
        }
        
        cant_see_ball_count_ = 0;
    }
    else
    {
        // EKF 不在跟踪（UNINIT）
        cant_see_ball_count_++;
        if (cant_see_ball_count_ * period_ms_ > 500)
            WM->set_ball_pos(Vector2d(0, 0), Vector2d(0, 0),
                             Vector2i(0, 0), 0, 0, false);
    }
    
    // 定位逻辑不变...
    if (localization_) { /* ... 原有代码不变 ... */ }
}
```

### 4.5 TCP 发送（L391-407 区域）

```cpp
if (OPTS->use_debug())
{
    // alpha/beta 发送：值已是 EKF 滤波后的 last_alpha_/last_beta_
    // ball_visible_ 已由 EKF 决定（含 COASTING 期间保持 true）
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

    // 图像发送代码不变...
}
```

## 5. 与 FSM 的协调关系

### 5.1 各场景时间线

**正常连续跟踪：**
```
Frame:   N    N+1   N+2   N+3   N+4   N+5   ...
ball_dets_: ●     ●     ●     ●     ●     ●
EKF:      TRK   TRK   TRK   TRK   TRK   TRK
can_see:  true  true  true  true  true  true
alpha/beta:  滤波后的平滑值
```
→ FSM 无感知，alpha/beta 比裸值更稳。

**短暂丢帧（球被遮挡 1-4 帧）：**
```
Frame:   N    N+1   N+2   N+3   N+4   N+5   N+6   ...
ball_dets_: ●     ○     ○     ○     ●     ●     ●
EKF:      TRK   COAST COAST COAST TRK   TRK   TRK
can_see:  true  true  true  true  true  true  true
```
→ FSM 全程看到 can_see=true。丢帧期间 alpha/beta 从 EKF 预测输出，球重新出现时用马氏门判断是否合理。**FSM 不会因 1-4 帧丢球误切回 SEARCH_BALL。**

**较长时间丢球（>4 帧）：**
```
Frame:   N    N+1..N+4  N+5   N+6   ...   N+15
ball_dets_: ●     ○       ○     ○            ○
EKF:      TRK   COAST    UNINIT
can_see:  true  true     false  (cant_see++ 开始计数)
                                      ...
                            500ms 后 WM set_ball_pos(false)
```
→ 200ms COASTING 后 EKF 主动放弃。ball_visible_=false，cant_see_ball_count_ 从 0 开始累加，500ms 后 WM 标记丢失。**总 worst-case 延迟：200ms + 500ms = 700ms（原有机制 500ms，多了 200ms）。**

### 5.2 FSM 各状态视角

| 状态 | 需要球？ | EKF 影响 |
|------|---------|---------|
| READY | 不需要 | 无影响 |
| SEARCH_BALL | 需要（找球） | COASTING 期间 can_see 保持 true，不会误触 "找到球" 逻辑——因为 COASTING 进入条件是 ball_dets_ 为空，而 SEARCH_BALL 的扫描靠 ball_dets_ 直接判断（见 scan_engine.cpp L101），不依赖 can_see |
| GOTO_BALL | 需要 | ball.alpha/beta 更稳，导航更平滑 |
| KICK_BALL | 需要（精细） | **最大收益**：alpha/beta 抗踏步抖动 + 丢帧不切走 |
| DRIBBLE | 需要 | 带球 alpha/beta 更稳，丢帧容忍 |
| SL | 不需要球 | 无影响 |
| GETUP | 不需要 | 无影响 |

### 5.3 关键：SEARCH_BALL 不受影响

SEARCH_BALL 中 `scan_engine.cpp:101` 直接读 `WM->ball()`，但扫描循环的判断是 `ball.can_see`（L102 `!ball.can_see`）。

COASTING 只在 **已经在 TRACKING 态且 ball_dets_ 变空** 时进入。SEARCH_BALL 里不会有 COASTING，因为：
- SEARCH_BALL 之前若球已丢失 >4 帧 → EKF 已是 UNINIT → is_tracking()=false → can_see=false
- SEARCH_BALL 中首次检测到球 → EKF 从 UNINIT reinit → 正常开始跟踪

**不会出现"没球但 EKF 假装有球导致 SEARCH_BALL 误判找到球"的 bug。**

## 6. 文件改动清单

| 文件 | 操作 | 行数 |
|------|------|------|
| `vision/ball_ekf.hpp` | 新增 | ~60 |
| `vision/ball_ekf.cpp` | 新增 | ~110 |
| `vision/CMakeLists.txt` | 修改 | +1 |
| `vision/vision.hpp` | 修改 | +4 |
| `vision/vision.cpp` | 修改，L290-320 重写 + updata() 加 IMU 晃动 + start() 初始化 | ~50 |

**不改的文件**：fsm.cpp、worldmodel.cpp/hpp、server.cpp、tcp.hpp、debuger 相关。

## 7. 调参方法

1. 先固定 `sigma_base=0.002, q_c=0.005, k_shake=1e-6, gate_thresh=9.0`
2. 录制一段 KICK_BALL 踏步微调时的数据（图像 + ball_dets + IMU）
3. 对比 EKF 输出 vs 裸值：
   - 踏步时 alpha/beta 跳动幅度减小 → 有效果
   - 球运动时延迟过大 → 增大 q_c 或减小 k_shake
   - 踏步时还不够平滑 → 增大 k_shake
4. COASTING 验证：在 KICK_BALL 中人工遮挡球 2-3 帧，确认 FSM 不切走
