# Post Flow - 即時人體姿態偵測

## 簡介
使用 YOLOv8 進行即時人體骨架偵測，支援 CUDA GPU 加速。

## 需求
- Python 3.11+
- NVIDIA GPU（選用）
- 攝影機

## 安裝

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境（Windows）
venv\Scripts\activate

# 安裝套件
pip install ultralytics torch torchvision opencv-python numpy
```

## 執行

```bash
python realtime_pose.py
```

第一次執行會自動下載 YOLOv8 模型（約 6MB）。

## 操作
- 按 `Q` 離開

## 關節點
使用 13 個主要關節點：

| 編號 | 名稱 |
|------|------|
| 0  | 鼻子 |
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

## GPU 加速
程式會自動偵測 CUDA，有 NVIDIA GPU 會自動使用。
