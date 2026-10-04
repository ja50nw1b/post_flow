# Post Flow - Real-time Human Pose Detection

## Introduction
Real-time human pose detection with two optimized versions:
- **Mobile/Edge**: Lightweight MediaPipe-only version for on-device, low-latency use
- **Server**: Pure YOLO GPU pipeline, optimized for high FPS on CUDA-capable machines

> **Note**: `pose_server.py` currently uses 2D joint angles. True 3D angle calculation via multi-camera triangulation is planned once multiple synchronized cameras are available. MediaPipe was removed from the server version because its CPU-only inference (~40ms) became the bottleneck, even with a GPU available for YOLO.

## Requirements
- Python 3.11+
- NVIDIA GPU (Optional, auto-detected for server version)
- Webcam

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install packages
pip install ultralytics torch torchvision opencv-python numpy mediapipe
```

## Files
| File | Target | Description |
|------|--------|-------------|
| `realtime_pose.py` | General | YOLO skeleton detection only (standalone or imported) |
| `pose_mobile.py` | Mobile / Edge devices | Pure MediaPipe Lite, CPU-only, low resolution for speed |
| `pose_server.py` | Server / GPU workstation | Pure YOLO GPU pipeline with 2D joint angles |

## Run

```bash
# Skeleton detection only
python realtime_pose.py

# Mobile/Edge version (lightweight, CPU)
python pose_mobile.py

# Server version (GPU-accelerated)
python pose_server.py
```

Models downloaded automatically on first run:
- YOLOv8l-pose (~87MB) — used by `realtime_pose.py` and `pose_server.py`
- MediaPipe Pose Landmarker Lite (~5MB) — used by `pose_mobile.py`

## Controls
- Press `Q` to quit

## Features
- Head position estimated using weighted average of eyes/ears, or spine-direction extrapolation
- `pose_mobile.py`: 3D joint angles using MediaPipe world coordinates (metres)
- `pose_server.py`: 2D joint angles, optimized for GPU throughput (3D triangulation planned)
- Occlusion handling with visibility threshold
- `pose_server.py` automatically uses CUDA if available, falls back to CPU otherwise

## Keypoints
| ID | Name |
|----|------|
| 5  | left_shoulder |
| 6  | right_shoulder |
| 7  | left_elbow |
| 8  | right_elbow |
| 9  | left_wrist |
| 10 | right_wrist |
| 11 | left_hip |
| 12 | right_hip |
| 13 | left_knee |
| 14 | right_knee |
| 15 | left_ankle |
| 16 | right_ankle |

## Roadmap
- [ ] Multi-camera 2D keypoint capture
- [ ] Camera calibration (intrinsics/extrinsics)
- [ ] Triangulation for true 3D coordinates in `pose_server.py`

## Mobile vs Server: Which to use?
| Scenario | Use |
|----------|-----|
| Running on a laptop/phone, battery life matters | `pose_mobile.py` |
| Running on a workstation/server with a GPU, need max FPS | `pose_server.py` |
| Just want skeleton overlay, no angle analysis | `realtime_pose.py` |