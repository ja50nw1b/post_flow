# Post Flow - Real-time Human Pose Detection

## Introduction
Real-time human pose detection using YOLOv8 with CUDA GPU support.

## Requirements
- Python 3.11+
- NVIDIA GPU (Optional)
- Webcam

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install packages
pip install ultralytics torch torchvision opencv-python numpy
```

## Run

```bash
python realtime_pose.py
```

The YOLOv8 model (~6MB) will be downloaded automatically on first run.

## Controls
- Press `Q` to quit

## Keypoints
Uses 13 main keypoints:

| ID | Name |
|----|------|
| 0  | nose |
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

## GPU Acceleration
CUDA is detected automatically. NVIDIA GPU will be used if available.
