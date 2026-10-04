# Post Flow - 即時人體姿態偵測

## 簡介
提供兩個優化版本的即時人體姿態偵測：
- **移動/邊緣裝置版**：純 MediaPipe 輕量版，適合裝置端、低延遲應用
- **伺服器版**：純 YOLO GPU 架構，針對 CUDA 裝置最佳化高 FPS

> **說明**：`pose_server.py` 目前使用 2D 關節角度。等未來有多台同步攝影機後，會改用三角測量計算真正的 3D 角度。MediaPipe 已從伺服器版本移除，因為即使 YOLO 有 GPU 加速，MediaPipe 的 CPU-only 推論（約 40ms）仍是效能瓶頸。

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
| `pose_server.py` | 伺服器/GPU 工作站 | 純 YOLO GPU 架構，搭配 2D 關節角度 |

## 執行

```bash
# 只跑骨架偵測
python realtime_pose.py

# 移動/邊緣裝置版（輕量，CPU）
python pose_mobile.py

# 伺服器版（GPU 加速）
python pose_server.py
```

第一次執行會自動下載模型：
- YOLOv8l-pose（約 87MB）— `realtime_pose.py` 與 `pose_server.py` 使用
- MediaPipe Pose Landmarker Lite（約 5MB）— `pose_mobile.py` 使用

## 操作
- 按 `Q` 離開

## 功能
- 頭部位置由眼睛/耳朵加權平均，或沿脊椎方向推算估算
- `pose_mobile.py`：使用 MediaPipe 世界座標（公尺單位）計算真實 3D 關節角度
- `pose_server.py`：2D 關節角度，針對 GPU 吞吐量最佳化（未來規劃 3D 三角測量）
- 可見度門檻處理遮擋問題
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

## 未來規劃
- [ ] 多機位 2D 關節點擷取
- [ ] 攝影機校正（內參/外參）
- [ ] `pose_server.py` 加入三角測量，計算真正的 3D 座標

## 該用移動版還是伺服器版？
| 情境 | 建議使用 |
|------|---------|
| 在筆電/手機上跑，考慮電池續航 | `pose_mobile.py` |
| 在有 GPU 的工作站/伺服器上跑，需要最高 FPS | `pose_server.py` |
| 只需要骨架疊加，不需要角度分析 | `realtime_pose.py` |