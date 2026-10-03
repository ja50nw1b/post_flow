# Post Flow - 即時人體姿態偵測

## 簡介
使用 YOLOv8l 進行即時人體骨架偵測，支援 CUDA GPU 加速。
使用 MediaPipe 世界座標進行精確的 3D 關節角度計算。

## 需求
- Python 3.11+
- NVIDIA GPU（選用，建議使用）
- 攝影機

## 安裝

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境（Windows）
venv\Scripts\activate

# 安裝套件
pip install ultralytics torch torchvision opencv-python numpy mediapipe
```

## 檔案說明
| 檔案 | 說明 |
|------|------|
| `realtime_pose.py` | 即時骨架偵測（可單獨執行） |
| `pose_3d.py` | 骨架偵測 + 3D 關節角度計算 |

## 執行

```bash
# 只跑骨架偵測
python realtime_pose.py

# 骨架 + 3D 角度偵測
python pose_3d.py
```

第一次執行會自動下載模型：
- YOLOv8l-pose（約 87MB）
- MediaPipe Pose Landmarker Heavy（約 30MB）

## 操作
- 按 `Q` 離開

## 功能
- 自動選擇運算裝置（CPU / CUDA GPU）
- 頭部位置由眼睛與耳朵加權平均估算
- 使用 MediaPipe 世界座標（公尺單位）計算真實 3D 關節角度
- 可見度門檻處理遮擋問題
- `pose_3d.py` 直接引用 `realtime_pose.py` 避免重複程式碼

## 關節點
| 編號 | 名稱 |
|------|------|
| 5  | 左肩 |
| 6  | 右肩 |
| 7  | 左肘 |
| 8  | 右肘 |
| 9  | 左腕 |
| 10 | 右腕 |
| 11 | 左髖 |
| 12 | 右髖 |
| 13 | 左膝 |
| 14 | 右膝 |
| 15 | 左踝 |
| 16 | 右踝 |

## 3D 角度偵測
`pose_3d.py` 使用 MediaPipe 3D 世界座標計算以下關節角度：

| 關節 | 使用的關節點 |
|------|------------|
| 左/右肩 | 頸中點 → 肩膀 → 手肘 |
| 左/右肘 | 肩膀 → 手肘 → 手腕 |
| 左/右髖 | 肩膀 → 髖部 → 膝蓋 |
| 左/右膝 | 髖部 → 膝蓋 → 腳踝 |

## GPU 加速
程式會自動偵測 CUDA，若有 NVIDIA GPU，使用者可以選擇是否使用 GPU 加速推論。