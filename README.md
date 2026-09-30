# Post Flow - Real-time Human Pose Detection

## Introduction
Real-time human pose detection using YOLOv8l with CUDA GPU support.
Features smooth head tracking using weighted average of nose, eyes and ears.

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
pip install ultralytics torch torchvision opencv-python numpy
```

## Run

```bash
python realtime_pose.py
```

The YOLOv8l model (~87MB) will be downloaded automatically on first run.

## Controls
- Press `Q` to quit

## Features
- Auto device selection (CPU / CUDA GPU)
- 13 main keypoints detection
- Head position estimated from nose + eyes + ears weighted average
- Moving average smoothing for stable head tracking
- Occlusion handling with visibility threshold

## Keypoints
Uses 13 main keypoints:

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

## Head Tracking
Head position is estimated using weighted average:

| Point | Weight |
|-------|--------|
| Nose | 0.1 |
| Left / Right Eye | 0.2 each |
| Left / Right Ear | 0.35 each |

## GPU Acceleration
CUDA is automatically detected. If a NVIDIA GPU is available, the user can choose to use it for faster inference.