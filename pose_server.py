# pose_server.py - 純 YOLO GPU 版本（等多機位設備到位後再加 3D 三角測量）
import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

# ── 裝置自動選擇（優先 GPU）───────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Using device: {device}")

# ── 初始化 YOLO ───────────────────────────────────────────
yolo = YOLO('yolov8l-pose.pt')
yolo.to(device)
if device == 'cpu':
    torch.set_num_threads(16)
print("✓ YOLO ready")

# ── 關節定義 ─────────────────────────────────────────────
YOLO_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
LEFT_EYE  = 1
RIGHT_EYE = 2
LEFT_EAR  = 3
RIGHT_EAR = 4

YOLO_CONNECTIONS = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

YOLO_COLORS = {
    (5,6):  (255,68,255),
    (5,7):  (255,191,0),  (7,9):  (255,191,0),
    (6,8):  (71,99,255),  (8,10): (71,99,255),
    (11,12):(150,255,150),
    (5,11): (150,255,150),(6,12): (150,255,150),
    (11,13):(255,113,218),(13,15):(255,113,218),
    (12,14):(0,140,255),  (14,16):(0,140,255),
}

VIS_THRESHOLD_DISPLAY = 0.5
VIS_THRESHOLD_MODEL   = 0.5
SKIP_IDS = [0, 1, 2, 3, 4]

# ── 工具函式 ─────────────────────────────────────────────
def to_pt(arr):
    return (int(arr[0]), int(arr[1]))

def calc_angle_2d(a, b, c):
    """2D 角度計算（暫時方案，等多機位 3D 三角測量完成後取代）"""
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def draw_angle(annotated, angle, pt, color=(0,255,0)):
    cv2.putText(annotated, f"{angle:.0f}",
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def estimate_head(pixel, vis, neck_px):
    """眼耳加權估算頭部位置"""
    pts, weights = [], []
    if vis[LEFT_EYE]  >= VIS_THRESHOLD_DISPLAY: pts.append(pixel[LEFT_EYE]);  weights.append(0.25)
    if vis[RIGHT_EYE] >= VIS_THRESHOLD_DISPLAY: pts.append(pixel[RIGHT_EYE]); weights.append(0.25)
    if vis[LEFT_EAR]  >= VIS_THRESHOLD_DISPLAY: pts.append(pixel[LEFT_EAR]);  weights.append(0.375)
    if vis[RIGHT_EAR] >= VIS_THRESHOLD_DISPLAY: pts.append(pixel[RIGHT_EAR]); weights.append(0.375)

    if pts:
        total_w     = sum(weights)
        head_center = sum(p * w for p, w in zip(pts, weights)) / total_w
        return head_center.astype(int)
    else:
        shoulder_dist = np.linalg.norm(pixel[5] - pixel[6])
        head_offset   = max(shoulder_dist * 0.8, 50)
        return np.array([int(neck_px[0]), int(neck_px[1] - head_offset)])

def detect_and_draw(frame):
    """YOLO 偵測 + 畫骨架 + 2D角度，回傳 annotated 圖和關節點資料（供未來三角測量用）"""
    results   = yolo(frame, verbose=False, conf=0.3)
    annotated = frame.copy()
    pixel     = None
    vis       = None

    for r in results:
        if r.keypoints is None: continue
        kps = r.keypoints.data
        if len(kps) == 0: continue

        kp    = kps[0].cpu().numpy()
        pixel = {i: kp[i, :2] for i in YOLO_IDS}
        vis   = {i: kp[i, 2]  for i in YOLO_IDS}

        neck_px = ((pixel[5] + pixel[6]) / 2).astype(int)
        hip_px  = ((pixel[11] + pixel[12]) / 2).astype(int)

        shoulder_ok = vis[5]  >= VIS_THRESHOLD_DISPLAY and vis[6]  >= VIS_THRESHOLD_DISPLAY
        hip_ok      = vis[11] >= VIS_THRESHOLD_DISPLAY and vis[12] >= VIS_THRESHOLD_DISPLAY

        # 頭部
        if shoulder_ok:
            head_px = estimate_head(pixel, vis, neck_px)
            cv2.line(annotated, to_pt(neck_px), to_pt(head_px),
                     (0,215,255), 3, cv2.LINE_AA)
            cv2.circle(annotated, to_pt(head_px), 18, (200,220,255), -1, cv2.LINE_AA)
            cv2.circle(annotated, to_pt(head_px), 18, (0,0,0), 1, cv2.LINE_AA)
            cv2.circle(annotated, to_pt(neck_px), 6, (255,255,255), -1)
            cv2.circle(annotated, to_pt(neck_px), 6, (0,0,0), 1)

        # 軀幹
        if shoulder_ok and hip_ok:
            cv2.line(annotated, to_pt(neck_px), to_pt(hip_px),
                     (150,255,150), 3, cv2.LINE_AA)

        # 四肢
        for i, j in YOLO_CONNECTIONS:
            if vis[i] < VIS_THRESHOLD_DISPLAY or vis[j] < VIS_THRESHOLD_DISPLAY:
                continue
            color = YOLO_COLORS.get((i,j), (180,180,180))
            cv2.line(annotated, to_pt(pixel[i]), to_pt(pixel[j]),
                     color, 3, cv2.LINE_AA)

        # 關節球
        for idx in YOLO_IDS:
            if idx in SKIP_IDS: continue
            dot_color = (100,100,100) if vis[idx] < VIS_THRESHOLD_DISPLAY \
                        else (255,255,255)
            cv2.circle(annotated, to_pt(pixel[idx]), 6, dot_color, -1, cv2.LINE_AA)
            cv2.circle(annotated, to_pt(pixel[idx]), 6, (0,0,0), 1, cv2.LINE_AA)

        if hip_ok:
            cv2.circle(annotated, to_pt(hip_px), 6, (255,255,255), -1)
            cv2.circle(annotated, to_pt(hip_px), 6, (0,0,0), 1)

        # ── 2D 角度計算（暫時方案）────────────────────
        angles = {}

        if vis[5] >= VIS_THRESHOLD_MODEL and vis[7] >= VIS_THRESHOLD_MODEL:
            angles['left_shoulder']  = (calc_angle_2d(neck_px, pixel[5], pixel[7]), pixel[5])
        if vis[6] >= VIS_THRESHOLD_MODEL and vis[8] >= VIS_THRESHOLD_MODEL:
            angles['right_shoulder'] = (calc_angle_2d(neck_px, pixel[6], pixel[8]), pixel[6])
        if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [5,7,9]):
            angles['left_elbow']     = (calc_angle_2d(pixel[5], pixel[7], pixel[9]), pixel[7])
        if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [6,8,10]):
            angles['right_elbow']    = (calc_angle_2d(pixel[6], pixel[8], pixel[10]), pixel[8])
        if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [5,11,13]):
            angles['left_hip']       = (calc_angle_2d(pixel[5], pixel[11], pixel[13]), pixel[11])
        if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [6,12,14]):
            angles['right_hip']      = (calc_angle_2d(pixel[6], pixel[12], pixel[14]), pixel[12])
        if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [11,13,15]):
            angles['left_knee']      = (calc_angle_2d(pixel[11], pixel[13], pixel[15]), pixel[13])
        if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [12,14,16]):
            angles['right_knee']     = (calc_angle_2d(pixel[12], pixel[14], pixel[16]), pixel[14])

        for name, (angle, pt) in angles.items():
            draw_angle(annotated, angle, to_pt(pt))

        break  # 只處理第一個偵測到的人

    return annotated, pixel, vis

# ── 主迴圈 ───────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_display = 0
    t_prev = time.time()

    print("✓ Streaming started, press Q to quit")
    print("⚠ Note: Currently using 2D angles. Multi-camera 3D triangulation coming soon.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, pixel, vis = detect_and_draw(frame)

        t_now       = time.time()
        fps_display = 0.9 * fps_display + 0.1 / (t_now - t_prev + 1e-6)
        t_prev      = t_now

        cv2.putText(annotated, f"FPS: {fps_display:.1f}  [{device.upper()}]",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow('Server Pose Detection (2D)  [Q to quit]', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()