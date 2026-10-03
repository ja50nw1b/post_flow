# Post Flow - Real-time Human Pose Detection

## Introduction
Real-time human pose detection using YOLOv8l with CUDA GPU support.
Features 3D joint angle calculation using MediaPipe world coordinates.

## Requirements
- Python 3.11+
- NVIDIA GPU (Optional but recommended)
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
| File | Description |
|------|-------------|
| `realtime_pose.py` | Real-time skeleton detection only |
| `pose_3d.py` | Skeleton detection + 3D joint angle calculation |

## Run

```bash
# Skeleton detection only
python realtime_pose.py

# Skeleton + 3D angle detection
python pose_3d.py
```

Models will be downloaded automatically on first run:
- YOLOv8l-pose (~87MB)
- MediaPipe Pose Landmarker Heavy (~30MB)

## Controls
- Press `Q` to quit

## Features
- Auto device selection (CPU / CUDA GPU)
- Head position estimated from eyes and ears weighted average
- 3D joint angles using MediaPipe world coordinates (metres)
- Occlusion handling with visibility threshold
- `pose_3d.py` imports from `realtime_pose.py` to avoid code duplication

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
`pose_3d.py` calculates the following joint angles using MediaPipe 3D world coordinates:

| Joint | Points Used |
|-------|------------|
| Left / Right Shoulder | Neck → Shoulder → Elbow |
| Left / Right Elbow | Shoulder → Elbow → Wrist |
| Left / Right Hip | Shoulder → Hip → Knee |
| Left / Right Knee | Hip → Knee → Ankle |

## GPU Acceleration
CUDA is automatically detected. If a NVIDIA GPU is available, the user can choose to use it for faster inference.