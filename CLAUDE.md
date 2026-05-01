# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SEU-UniRobot -- Southeast University RoboCup KidSize humanoid robot controller. C++11/CUDA project targeting NVIDIA Jetson Orin NX for autonomous humanoid soccer robots. Currently at V2.1.

## Build Commands

```bash
# x86_64 native build (development/debug)
python3 x86_64-build.py
# Binaries: bin/x86_64/controller

# aarch64 cross-compile for Jetson Orin NX
python3 aarch64-build.py
# Binaries: bin/aarch64/controller

# Deploy to robot (build + SSH upload + run)
python3 src/scripts/start_robot.py
```

Run with `-h` for CLI flags: `-p` player ID, `-d` debug, `-c` camera, `-r` robot, `-g` gamectrl, `-s` comm, `-k` kick mode, `-m` remote, `-f` control flag (0=auto, 1=keyboard), `-i` image record.

## Dependencies

- CUDA >= 9.0, cuDNN >= 7.0, TensorRT (at `/usr/local/tensorrt`)
- OpenCV >= 3.3.1, Eigen3, Boost (system, program_options, asio, filesystem)
- Qt5 (x86_64 debug tools only), OpenGL/GLUT/GLU, libv4l-dev
- ZED SDK (optional, auto-detected at `/usr/local/zed/`)
- Python3: paramiko, transitions, ssh2-python (deployment scripts)

## Architecture

### Entry & Initialization

`src/controller/main.cpp` -> creates `Player` ("maxwell") -> `OPTS->init()` -> `CONF->init()` -> `ROBOT->init()` -> `Player::init()`

### Core Patterns

- **Singletons**: `Configuration` (CONF), `Options` (OPTS), `WorldModel` (WM), `Vision` (VISION), `WalkEngine` (WE), `ScanEngine` (SE), `ActionEngine` (AE), `LedEngine` (LE), `Localization` (SL), `Robot` (ROBOT)
- **Observer (Pub/Sub)**: Sensors (Camera, IMU, Motor, Button, GameCtrl, Hear) are Publishers; WorldModel, Vision, WalkEngine are Subscribers
- **FSM**: Player behavior driven by finite state machine: READY, GETUP, SEARCH_BALL, GOTO_BALL, KICK_BALL, DRIBBLE, SL
- **Task**: Each think cycle produces Tasks (WalkTask, ActionTask, LookTask, GcretTask, SayTask, LedTask) executed sequentially
- **Timer**: Player, Vision, and engines run periodic loops via Timer base class

### Data Flow

Camera capture -> Vision pipeline (GPU color conversion, resize, undistortion, letterbox, TensorRT YOLOv8 inference) -> detections update WorldModel -> Player FSM decides actions -> Tasks drive engines -> Motor sensor sends Dynamixel joint commands

### Key Subsystems

| Subsystem | Path | Purpose |
|---|---|---|
| Player | `src/controller/player/` | Central orchestrator, FSM, think loop |
| Vision | `src/controller/player/vision/` | YOLOv8 TensorRT 10 detector, GPU image pipeline |
| Sensors | `src/controller/player/sensor/` | Camera (V4L2/ZED), IMU, Motors, GameCtrl, Button, Hear |
| Engines | `src/controller/player/engine/` | Walk (IKWalk), Action, Scan, LED |
| Localization | `src/controller/player/localization/` | Kalman filter self-localization |
| WorldModel | `src/controller/player/core/worldmodel.hpp` | Central world state singleton |
| Server | `src/controller/player/server/` | TCP debug server (Boost.Asio) |
| Debug Tools | `src/tools/debuger/` | Qt5 GUI: image monitor, action debugger, walk remote, etc. (x86_64 only) |

### Configuration

All behavior is config-driven via JSON files in `src/data/`:
- `config.conf` -- master config (team info, strategy positions, thresholds, per-player settings)
- `model/robot.conf` -- kinematic tree, joint hierarchy, bone lengths
- `action/*.conf` -- walk parameters, action sequences, scan configs, joint offsets
- `algorithm/*.engine` -- TensorRT engine files for ball/post detection

### Threading

Each major component runs in its own thread: camera capture, vision processing, walk engine, scan engine, action engine, LED engine, motor communication, IMU polling, game controller listener, TCP server, manual control input.

### Platform Support

Dual-platform: x86_64 (development with Qt5 debug tools) and aarch64/Jetson (deployment). CMake `CROSS` flag controls target. ZED-mini backend compiled conditionally via `USE_ZED_BACKEND` define.

### Vendored Code

- Dynamixel SDK (`src/controller/drivers/dynamixel/`)
- Rhoban/Leph IKWalk (`src/controller/player/engine/walk/`)
- RoboCup Game Controller headers (`src/lib/udp_data/`)
- HipNUC IMU decoder (in sensor sources)
