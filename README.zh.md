# Post Flow - 即時人體姿態偵測

## 簡介
提供兩個優化版本的即時人體姿態偵測：
- **移動/邊緣裝置版**：純 MediaPipe 輕量版，適合裝置端、低延遲應用
- **伺服器版**：YOLO + MediaPipe 混合架構，CUDA GPU 加速，精度優先

## 需求
- Python 3.11+
- NVIDIA GPU（選用，伺服器版會自動偵測）
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
| 檔案 | 目標環境 | 說明 |
|------|---------|------|
| `realtime_pose.py` | 通用 | 純 YOLO 骨架偵測（可獨立執行或被引用） |
| `pose_mobile.py` | 手機/邊緣裝置 | 純 MediaPipe Lite，CPU 執行，低解析度換取速度 |
| `pose_server.py` | 伺服器/GPU 工作站 | YOLO（GPU）+ MediaPipe 3D 角度，自動偵測 CUDA |

## 執行

```bash
# 只跑骨架偵測
python realtime_pose.py

# 移動/邊緣裝置版（輕量，CPU）
python pose_mobile.py

# 伺服器版（GPU 加速，高精度）
python pose_server.py
```

第一次執行會自動下載模型：
- YOLOv8l-pose（約 87MB）— `realtime_pose.py` 與 `pose_server.py` 使用
- MediaPipe Pose Landmarker Lite（約 5MB）— `pose_mobile.py` 使用
- MediaPipe Pose Landmarker Heavy（約 30MB）— `pose_server.py` 使用

## 操作
- 按 `Q` 離開

## 功能
- 頭部位置由眼睛/耳朵加權平均，或沿脊椎方向推算估算
- 使用 MediaPipe 世界座標（公尺單位）計算真實 3D 關節角度
- 可見度門檻處理遮擋問題
- `pose_server.py` 直接引用 `realtime_pose.py` 的共用函式，避免重複程式碼
- `pose_server.py` 自動偵測並使用 CUDA，沒有的話自動退回 CPU

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
`pose_mobile.py` 與 `pose_server.py` 皆使用 MediaPipe 3D 世界座標計算以下關節角度：

| 關節 | 使用的關節點 |
|------|------------|
| 左/右肩 | 頸中點 → 肩膀 → 手肘 |
| 左/右肘 | 肩膀 → 手肘 → 手腕 |
| 左/右髖 | 肩膀 → 髖部 → 膝蓋 |
| 左/右膝 | 髖部 → 膝蓋 → 腳踝 |

## 該用移動版還是伺服器版？
| 情境 | 建議使用 |
|------|---------|
| 在筆電/手機上跑，考慮電池續航 | `pose_mobile.py` |
| 在有 GPU 的工作站/伺服器上跑，精度優先 | `pose_server.py` |
| 只需要骨架疊加，不需要角度分析 | `realtime_pose.py` |