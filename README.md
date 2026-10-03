# Post Flow - Real-time Human Pose Detection

## Introduction
Real-time human pose detection with two optimized versions:
- **Mobile/Edge**: Lightweight MediaPipe-only version for on-device, low-latency use
- **Server**: YOLO + MediaPipe hybrid with CUDA GPU acceleration for high accuracy

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
| `pose_server.py` | Server / GPU workstation | YOLO (GPU) + MediaPipe 3D angles, auto-detects CUDA |

## Run

```bash
# Skeleton detection only
python realtime_pose.py

# Mobile/Edge version (lightweight, CPU)
python pose_mobile.py

# Server version (GPU-accelerated, high accuracy)
python pose_server.py
```

Models downloaded automatically on first run:
- YOLOv8l-pose (~87MB) — used by `realtime_pose.py` and `pose_server.py`
- MediaPipe Pose Landmarker Lite (~5MB) — used by `pose_mobile.py`
- MediaPipe Pose Landmarker Heavy (~30MB) — used by `pose_server.py`

## Controls
- Press `Q` to quit

## Features
- Head position estimated using weighted average of eyes/ears, or spine-direction extrapolation
- 3D joint angles using MediaPipe world coordinates (metres)
- Occlusion handling with visibility threshold
- `pose_server.py` imports shared utilities from `realtime_pose.py` to avoid code duplication
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

## 3D Angle Detection
Both `pose_mobile.py` and `pose_server.py` calculate the following joint angles using MediaPipe 3D world coordinates:

| Joint | Points Used |
|-------|------------|
| Left / Right Shoulder | Neck → Shoulder → Elbow |
| Left / Right Elbow | Shoulder → Elbow → Wrist |
| Left / Right Hip | Shoulder → Hip → Knee |
| Left / Right Knee | Hip → Knee → Ankle |

## Mobile vs Server: Which to use?
| Scenario | Use |
|----------|-----|
| Running on a laptop/phone, battery life matters | `pose_mobile.py` |
| Running on a workstation/server with a GPU, accuracy matters most | `pose_server.py` |
| Just want skeleton overlay, no angle analysis | `realtime_pose.py` |