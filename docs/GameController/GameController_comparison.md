# GameController 数据结构对比分析

本文档对比当前项目 (`src/lib/udp_data/RoboCupGameControlData.h`) 与最新 GameController (`/home/ph/GameController/game_controller_msgs/headers/RoboCupGameControlData.h`) 的差异。

## 一、版本号变化

| 项目 | 旧版本 (当前项目) | 新版本 (GameController) |
|------|------------------|------------------------|
| GAMECONTROLLER_STRUCT_VERSION | **12** | **19** |
| GAMECONTROLLER_RETURN_STRUCT_VERSION | **2** | **4** |
| MAX_NUM_PLAYERS | **11** | **20** |

---

## 二、RoboCupGameControlData 结构变化

### 字段对比表

| 字段 | 旧版本类型 | 新版本类型 | 变化说明 |
|------|-----------|-----------|---------|
| `header[4]` | char | char | 不变 |
| `version` | **uint16_t** | **uint8_t** | 类型缩小 |
| `packetNumber` | uint8_t | uint8_t | 不变 |
| `playersPerTeam` | uint8_t | uint8_t | 不变 |
| `gameType` → `competitionType` | uint8_t | uint8_t | 重命名 |
| `stopped` | - | **uint8_t** | **新增**：比赛是否暂停 |
| `secondaryState` → `gamePhase` | uint8_t | uint8_t | 重命名，语义调整 |
| `state` | uint8_t | uint8_t | 不变 |
| `setPlay` | - | **uint8_t** | **新增**：任意球类型 |
| `firstHalf` | uint8_t | uint8_t | 不变 |
| `kickOffTeam` → `kickingTeam` | uint8_t | uint8_t | 重命名，语义扩展 |
| `secondaryStateInfo[4]` | char | - | **删除** |
| `dropInTeam` | uint8_t | - | **删除** |
| `dropInTime` | uint16_t | - | **删除** |
| `secsRemaining` | **uint16_t** | **int16_t** | 类型变化（支持负值） |
| `secondaryTime` | **uint16_t** | **int16_t** | 类型变化（支持负值） |
| `teams[2]` | TeamInfo | TeamInfo | 结构变化（见下节） |

---

## 三、TeamInfo 结构变化

### 字段对比表

| 字段 | 旧版本 | 新版本 | 变化说明 |
|------|-------|-------|---------|
| `teamNumber` | uint8_t | uint8_t | 不变 |
| `teamColour` | uint8_t | - | **删除**，拆分为两字段 |
| `fieldPlayerColour` | - | **uint8_t** | **新增**：场上球员颜色 |
| `goalkeeperColour` | - | **uint8_t** | **新增**：守门员颜色 |
| `goalkeeper` | - | **uint8_t** | **新增**：守门员球员号码 |
| `score` | uint8_t | uint8_t | 不变 |
| `penaltyShot` | uint8_t | uint8_t | 不变 |
| `singleShots` | uint16_t | uint16_t | 不变 |
| `messageBudget` | - | **uint16_t** | **新增**：队伍消息预算 |
| `coachSequence` | uint8_t | - | **删除** |
| `coachMessage[253]` | uint8_t[] | - | **删除** |
| `coach` | RobotInfo | - | **删除** |
| `players[]` | RobotInfo[11] | RobotInfo[20] | 数量增加 |

---

## 四、RobotInfo 结构变化

| 字段 | 旧版本 | 新版本 | 变化说明 |
|------|-------|-------|---------|
| `penalty` | uint8_t | uint8_t | 不变（但罚球码变化，见下节） |
| `secsTillUnpenalised` | uint8_t | uint8_t | 不变 |
| `yellowCardCount` | uint8_t | - | **删除** |
| `redCardCount` | uint8_t | - | **删除** |
| `warnings` | - | **uint8_t** | **新增**：警告次数 |
| `cautions` | - | **uint8_t** | **新增**：黄牌次数 |

**语义变化**: 旧版 `yellowCardCount` 对应新版 `cautions`，但新版新增了 `warnings` 作为更轻的处罚记录。

---

## 五、状态/阶段定义变化

### 旧版 secondaryState（已废弃）
```c
#define STATE2_NORMAL               0
#define STATE2_PENALTYSHOOT         1
#define STATE2_OVERTIME             2
#define STATE2_TIMEOUT              3
#define STATE2_DIRECT_FREEKICK      4
#define STATE2_INDIRECT_FREEKICK    5
#define STATE2_PENALTYKICK          6
```

### 新版 gamePhase + setPlay（分离设计）
```c
// gamePhase - 比赛阶段
#define GAME_PHASE_NORMAL            0
#define GAME_PHASE_PENALTY_SHOOT_OUT 1
#define GAME_PHASE_EXTRA_TIME        2
#define GAME_PHASE_TIMEOUT           3

// setPlay - 任意球/定位球类型
#define SET_PLAY_NONE               0
#define SET_PLAY_DIRECT_FREE_KICK   1
#define SET_PLAY_INDIRECT_FREE_KICK 2
#define SET_PLAY_PENALTY_KICK       3
#define SET_PLAY_THROW_IN           4   // 新增
#define SET_PLAY_GOAL_KICK          5   // 新增
#define SET_PLAY_CORNER_KICK        6   // 新增
```

**关键变化**: 旧版将任意球状态混在 `secondaryState` 中，新版将其分离为独立的 `setPlay` 字段，并新增了三种定位球类型。

---

## 六、罚球码变化

### 旧版罚球码（已废弃）
```c
// SPL 罚球码
#define SPL_ILLEGAL_BALL_CONTACT    1
#define SPL_PLAYER_PUSHING          2
#define SPL_ILLEGAL_MOTION_IN_SET   3
#define SPL_INACTIVE_PLAYER         4
#define SPL_ILLEGAL_DEFENDER        5
#define SPL_LEAVING_THE_FIELD       6
#define SPL_KICK_OFF_GOAL           7
#define SPL_REQUEST_FOR_PICKUP      8
#define SPL_COACH_MOTION            9

// HL 罚球码
#define HL_BALL_MANIPULATION        30
#define HL_PHYSICAL_CONTACT         31
#define HL_ILLEGAL_ATTACK           32
#define HL_ILLEGAL_DEFENSE          33
#define HL_PICKUP_OR_INCAPABLE      34
#define HL_SERVICE                  35

#define SUBSTITUTE                  14
#define MANUAL                      15
```

### 新版罚球码（统一编码）
```c
#define PENALTY_NONE                          0
#define PENALTY_ILLEGAL_POSITIONING           1   // 新增
#define PENALTY_MOTION_IN_SET                 2   // 对应旧 SPL_ILLEGAL_MOTION_IN_SET
#define PENALTY_LOCAL_GAME_STUCK              3   // 新增
#define PENALTY_INCAPABLE_ROBOT               4   // 对应旧 HL_PICKUP_OR_INCAPABLE 部分
#define PENALTY_PICK_UP                       5   // 对应旧 SPL_REQUEST_FOR_PICKUP
#define PENALTY_BALL_HOLDING                  6   // 对应旧 HL_BALL_MANIPULATION
#define PENALTY_LEAVING_THE_FIELD             7   // 对应旧 SPL_LEAVING_THE_FIELD
#define PENALTY_PLAYING_WITH_ARMS_HANDS       8   // 新增
#define PENALTY_PUSHING                       9   // 对应旧 SPL_PLAYER_PUSHING / HL_PHYSICAL_CONTACT
#define PENALTY_SENT_OFF                      10  // 新增
#define PENALTY_SUBSTITUTE                    11  // 旧版为 14
```

**关键变化**:
1. 罚球码统一为连续编码 0-11，不再区分 SPL/HL 两套码
2. `HL_PICKUP_OR_INCAPABLE` (34) 拆分为 `PENALTY_INCAPABLE_ROBOT` (4) 和 `PENALTY_PICK_UP` (5)
3. 新增 `PENALTY_ILLEGAL_POSITIONING`, `PENALTY_LOCAL_GAME_STUCK`, `PENALTY_PLAYING_WITH_ARMS_HANDS`, `PENALTY_SENT_OFF`

---

## 七、比赛类型变化

| 旧版 | 新版 | 说明 |
|-----|-----|-----|
| `GAME_ROUNDROBIN = 0` | `COMPETITION_TYPE_SMALL = 0` | 小型联赛 |
| `GAME_PLAYOFF = 1` | `COMPETITION_TYPE_MIDDLE = 1` | 中型联赛 |
| `GAME_DROPIN = 2` | `COMPETITION_TYPE_LARGE = 2` | 大型联赛 |

---

## 八、踢球方变化

| 旧版 | 新版 | 说明 |
|-----|-----|-----|
| `kickOffTeam` | `kickingTeam` | 重命名，语义扩展为任意球踢球方 |
| `DROPBALL = 255` | `KICKING_TEAM_NONE = 255` | 重命名，表示无踢球方 |

---

## 九、RoboCupGameControlReturnData 变化（机器人返回消息）

### 旧版结构
```c
struct RoboCupGameControlReturnData {
    char header[4];           // "RGrt"
    uint8_t version;          // 2
    uint8_t team;             // 队伍编号
    uint8_t player;           // 球员编号 (1-...)
    uint8_t message;          // 消息类型
};
// 消息类型:
#define GAMECONTROLLER_RETURN_MSG_MAN_PENALISE   0
#define GAMECONTROLLER_RETURN_MSG_MAN_UNPENALISE 1
#define GAMECONTROLLER_RETURN_MSG_ALIVE          2
```

### 新版结构
```c
struct RoboCupGameControlReturnData {
    char header[4];           // "RGrt"
    uint8_t version;          // 4
    uint8_t playerNum;        // 球员编号
    uint8_t teamNum;          // 队伍编号
    uint8_t fallen;           // 是否跌倒
    float pose[3];            // x, y, theta (位置和朝向)
    float ballAge;            // 最后看到球的时间
    float ball[2];            // 相对于球的距离
};
```

**关键变化**:
1. **删除了 `message` 字段**：机器人不再通过返回消息请求处罚/解罚
2. **新增位置和球信息**：机器人现在需要报告自身位置和球的信息
3. **新增 `fallen` 字段**：报告跌倒状态
4. 字段顺序变化：`player` → `team` 改为 `playerNum` → `teamNum`

---

## 十、当前项目代码需要修改的部分

### 1. `src/lib/udp_data/RoboCupGameControlData.h`
需要完全替换为新版结构定义。

### 2. `src/controller/player/sensor/gamectrl.hpp`
枚举定义需要更新：
```cpp
// 需要删除的枚举
enum RobocupGameSecondaryState { ... };  // 改用 gamePhase + setPlay

// 需要更新的枚举
enum RobocupPlayerState { ... };  // 罚球码全部变化
```

### 3. `src/controller/player/sensor/gamectrl.cpp`
接收逻辑需要适配新版结构大小。

### 4. `src/controller/player/play_with_gc.cpp`
需要修改：
- `gc_data.secondaryState` → `gc_data.gamePhase` + `gc_data.setPlay`
- `gc_data.kickOffTeam` → `gc_data.kickingTeam`
- `gc_data.secondaryTime` 类型从 `uint16_t` 变为 `int16_t`
- 罚球判断条件：`HL_PICKUP_OR_INCAPABLE` (34) → `PENALTY_INCAPABLE_ROBOT` (4) 或 `PENALTY_PICK_UP` (5)
- `gc_data.teams[team_index].players[id].penalty` 的值判断需要全部更新

### 5. `src/controller/player/core/worldmodel.cpp`
如果使用了 `secondaryState`、罚球码等，需要更新。

### 6. TeamInfo 相关代码
- `gc_data.teams[i].teamColour` → `gc_data.teams[i].fieldPlayerColour` / `goalkeeperColour`
- 删除对 `coachMessage`、`coach` 的引用

---

## 十一、兼容性风险点

1. **结构体大小变化**：新版结构体大小不同，直接接收会导致数据错位
2. **版本号校验**：新版 GameController 可能拒绝旧版 version=12 的消息
3. **罚球码语义变化**：旧代码 `penalty == 34` 的判断在新版中无意义
4. **字段删除**：访问 `dropInTeam`、`secondaryStateInfo` 等字段会导致编译错误
5. **返回消息格式变化**：如果机器人需要发送状态消息，需要按新版格式构造

---

## 十二、建议迁移步骤

1. 替换 `RoboCupGameControlData.h` 为新版定义
2. 更新 `gamectrl.hpp` 中的枚举定义
3. 修改 `play_with_gc.cpp` 中所有状态判断逻辑
4. 测试接收新版消息的正确性（验证 header + version）
5. 如需发送返回消息，实现新版 `RoboCupGameControlReturnData` 构造逻辑