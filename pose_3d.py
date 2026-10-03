# pose_3d.py - YOLO 骨架 + MediaPipe 3D 角度
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os

# ── import realtime_pose 的共用部分 ──────────────────────
from realtime_pose import (
    device,
    VIS_THRESHOLD_MODEL,
    to_pt,
    draw_angle,
    detect_and_draw,
)

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

# ── 3D 角度計算 ───────────────────────────────────────────
def calc_angle_3d(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def calc_3d_angles(annotated, frame):
    """MediaPipe 3D 角度計算並顯示"""
    img_h, img_w = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms    = int(time.time() * 1000)
    result   = mp_detector.detect_for_video(mp_image, ts_ms)

    if not result.pose_landmarks or not result.pose_world_landmarks:
        return

    lm  = result.pose_landmarks[0]
    wlm = result.pose_world_landmarks[0]
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

        # YOLO 骨架
        annotated, _, _ = detect_and_draw(frame)

        # MediaPipe 3D 角度
        calc_3d_angles(annotated, frame)

        t_now       = time.time()
        fps_display = 0.9 * fps_display + 0.1 / (t_now - t_prev + 1e-6)
        t_prev      = t_now

        cv2.putText(annotated,
                    f"FPS: {fps_display:.1f}  [{device.upper()} | MediaPipe 3D]",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow('3D Pose Detection  [Q to quit]', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    mp_detector.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()