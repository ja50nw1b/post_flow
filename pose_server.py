# pose_server.py - YOLO 骨架 + MediaPipe 3D 角度（伺服器版，自動用 GPU）
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO
import torch
import urllib.request
import os

# ── 下載 MediaPipe 模型 ───────────────────────────────────
MODEL_PATH = 'pose_landmarker_heavy.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
        MODEL_PATH
    )
    print("✓ MediaPipe model downloaded")

# ── 裝置自動選擇（優先 GPU）───────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Using device: {device}")

# ── 初始化 YOLO ───────────────────────────────────────────
yolo = YOLO('yolov8l-pose.pt')
yolo.to(device)
if device == 'cpu':
    torch.set_num_threads(16)
print("✓ YOLO ready")

# ── 初始化 MediaPipe ──────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO,
)
mp_detector = mp_vision.PoseLandmarker.create_from_options(options)
print("✓ MediaPipe 3D ready")

# ── YOLO 關節定義 ─────────────────────────────────────────
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

def calc_angle_3d(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def draw_angle(annotated, angle, pt, color=(0,255,0)):
    cv2.putText(annotated, f"{angle:.0f}",
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def estimate_head_yolo(pixel, vis, neck_px):
    """YOLO 眼耳加權估算頭部"""
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

# ── 主迴圈 ───────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_display = 0
    t_prev = time.time()

    print("✓ Streaming started, press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        annotated    = frame.copy()

        # ── YOLO 推論（骨架顯示）────────────────────────
        yolo_results = yolo(frame, verbose=False, conf=0.3)

        for r in yolo_results:
            if r.keypoints is None: continue
            kps = r.keypoints.data
            if len(kps) == 0: continue

            kp         = kps[0].cpu().numpy()
            yolo_pixel = {i: kp[i, :2] for i in YOLO_IDS}
            yolo_vis   = {i: kp[i, 2]  for i in YOLO_IDS}

            neck_px = ((yolo_pixel[5] + yolo_pixel[6]) / 2).astype(int)
            hip_px  = ((yolo_pixel[11] + yolo_pixel[12]) / 2).astype(int)

            shoulder_ok = yolo_vis[5]  >= VIS_THRESHOLD_DISPLAY and yolo_vis[6]  >= VIS_THRESHOLD_DISPLAY
            hip_ok      = yolo_vis[11] >= VIS_THRESHOLD_DISPLAY and yolo_vis[12] >= VIS_THRESHOLD_DISPLAY

            if shoulder_ok:
                head_px = estimate_head_yolo(yolo_pixel, yolo_vis, neck_px)
                cv2.line(annotated, to_pt(neck_px), to_pt(head_px),
                         (0,215,255), 3, cv2.LINE_AA)
                cv2.circle(annotated, to_pt(head_px), 18, (200,220,255), -1, cv2.LINE_AA)
                cv2.circle(annotated, to_pt(head_px), 18, (0,0,0), 1, cv2.LINE_AA)
                cv2.circle(annotated, to_pt(neck_px), 6, (255,255,255), -1)
                cv2.circle(annotated, to_pt(neck_px), 6, (0,0,0), 1)

            if shoulder_ok and hip_ok:
                cv2.line(annotated, to_pt(neck_px), to_pt(hip_px),
                         (150,255,150), 3, cv2.LINE_AA)

            for i, j in YOLO_CONNECTIONS:
                if yolo_vis[i] < VIS_THRESHOLD_DISPLAY or yolo_vis[j] < VIS_THRESHOLD_DISPLAY:
                    continue
                color = YOLO_COLORS.get((i,j), (180,180,180))
                cv2.line(annotated, to_pt(yolo_pixel[i]), to_pt(yolo_pixel[j]),
                         color, 3, cv2.LINE_AA)

            for idx in YOLO_IDS:
                if idx in SKIP_IDS: continue
                dot_color = (100,100,100) if yolo_vis[idx] < VIS_THRESHOLD_DISPLAY \
                            else (255,255,255)
                cv2.circle(annotated, to_pt(yolo_pixel[idx]), 6, dot_color, -1, cv2.LINE_AA)
                cv2.circle(annotated, to_pt(yolo_pixel[idx]), 6, (0,0,0), 1, cv2.LINE_AA)

            if hip_ok:
                cv2.circle(annotated, to_pt(hip_px), 6, (255,255,255), -1)
                cv2.circle(annotated, to_pt(hip_px), 6, (0,0,0), 1)

        # ── MediaPipe 推論（3D 角度）────────────────────
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms     = int(time.time() * 1000)
        mp_result = mp_detector.detect_for_video(mp_image, ts_ms)

        if mp_result.pose_landmarks and mp_result.pose_world_landmarks:
            lm  = mp_result.pose_landmarks[0]
            wlm = mp_result.pose_world_landmarks[0]
            vis = [l.visibility for l in lm]

            def w3(idx):
                return [wlm[idx].x, wlm[idx].y, wlm[idx].z]

            def neck_w3():
                return [(wlm[11].x+wlm[12].x)/2,
                        (wlm[11].y+wlm[12].y)/2,
                        (wlm[11].z+wlm[12].z)/2]

            def mp_pt(idx):
                return (int(lm[idx].x * img_w), int(lm[idx].y * img_h))

            angles = {}

            if vis[11] >= VIS_THRESHOLD_MODEL and vis[13] >= VIS_THRESHOLD_MODEL:
                angles['left_shoulder']  = (calc_angle_3d(neck_w3(), w3(11), w3(13)), mp_pt(11))
            if vis[12] >= VIS_THRESHOLD_MODEL and vis[14] >= VIS_THRESHOLD_MODEL:
                angles['right_shoulder'] = (calc_angle_3d(neck_w3(), w3(12), w3(14)), mp_pt(12))
            if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [11,13,15]):
                angles['left_elbow']     = (calc_angle_3d(w3(11), w3(13), w3(15)), mp_pt(13))
            if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [12,14,16]):
                angles['right_elbow']    = (calc_angle_3d(w3(12), w3(14), w3(16)), mp_pt(14))
            if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [11,23,25]):
                angles['left_hip']       = (calc_angle_3d(w3(11), w3(23), w3(25)), mp_pt(23))
            if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [12,24,26]):
                angles['right_hip']      = (calc_angle_3d(w3(12), w3(24), w3(26)), mp_pt(24))
            if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [23,25,27]):
                angles['left_knee']      = (calc_angle_3d(w3(23), w3(25), w3(27)), mp_pt(25))
            if all(vis[i] >= VIS_THRESHOLD_MODEL for i in [24,26,28]):
                angles['right_knee']     = (calc_angle_3d(w3(24), w3(26), w3(28)), mp_pt(26))

            for name, (angle, pt) in angles.items():
                draw_angle(annotated, angle, pt)

        # FPS
        t_now       = time.time()
        fps_display = 0.9 * fps_display + 0.1 / (t_now - t_prev + 1e-6)
        t_prev      = t_now

        cv2.putText(annotated, f"FPS: {fps_display:.1f}  [{device.upper()} | MediaPipe 3D]",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow('Server Pose Detection  [Q to quit]', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    mp_detector.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()