# pose_mobile.py - 純 MediaPipe 版本（移動端）
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os

MODEL_PATH = 'pose_landmarker_lite.task'  # 用 lite 版，更適合移動端
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe lite model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
        MODEL_PATH
    )
    print("✓ Model downloaded")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO,
)
detector = mp_vision.PoseLandmarker.create_from_options(options)
print("✓ MediaPipe Lite ready (Mobile-optimized)")

CONNECTIONS = [
    (11,12), (11,13), (13,15),
    (12,14), (14,16),
    (11,23), (12,24), (23,24),
    (23,25), (25,27),
    (24,26), (26,28),
]

COLORS = {
    (11,12):(255,68,255),
    (11,13):(255,191,0),  (13,15):(255,191,0),
    (12,14):(71,99,255),  (14,16):(71,99,255),
    (11,23):(150,255,150),(12,24):(150,255,150),
    (23,24):(150,255,150),
    (23,25):(255,113,218),(25,27):(255,113,218),
    (24,26):(0,140,255),  (26,28):(0,140,255),
}

VIS_THRESHOLD = 0.5
SKIP_IDS = {0,1,2,3,4,5,6,7,8,9,10}

def to_pt(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def calc_angle_3d(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def draw_angle(img, angle, pt, color=(0,255,0)):
    cv2.putText(img, f"{angle:.0f}", (pt[0]+10, pt[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def estimate_head(lm, neck_px, w, h):
    hip_x = (lm[23].x + lm[24].x) / 2 * w
    hip_y = (lm[23].y + lm[24].y) / 2 * h
    dx, dy = neck_px[0]-hip_x, neck_px[1]-hip_y
    spine_len = np.sqrt(dx**2+dy**2)
    if spine_len < 1e-6:
        return (neck_px[0], int(neck_px[1]-50))
    sd = np.sqrt((lm[11].x-lm[12].x)**2+(lm[11].y-lm[12].y)**2) * w
    off = max(sd*0.8, 50)
    return (int(neck_px[0]+dx/spine_len*off), int(neck_px[1]+dy/spine_len*off))

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 移動端用較小解析度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_d, t_prev = 0, time.time()
    print("✓ Streaming started, press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time()*1000)
        result = detector.detect_for_video(mp_img, ts)

        annotated = frame.copy()

        if result.pose_landmarks and result.pose_world_landmarks:
            lm, wlm = result.pose_landmarks[0], result.pose_world_landmarks[0]
            vis = [l.visibility for l in lm]

            neck_px = (int((lm[11].x+lm[12].x)/2*w), int((lm[11].y+lm[12].y)/2*h))
            hip_px  = (int((lm[23].x+lm[24].x)/2*w), int((lm[23].y+lm[24].y)/2*h))

            sh_ok = vis[11]>=VIS_THRESHOLD and vis[12]>=VIS_THRESHOLD
            hp_ok = vis[23]>=VIS_THRESHOLD and vis[24]>=VIS_THRESHOLD

            if sh_ok and hp_ok:
                head_px = estimate_head(lm, neck_px, w, h)
                cv2.line(annotated, neck_px, head_px, (0,215,255), 3, cv2.LINE_AA)
                cv2.circle(annotated, head_px, 18, (200,220,255), -1, cv2.LINE_AA)
                cv2.circle(annotated, head_px, 18, (0,0,0), 1, cv2.LINE_AA)
                cv2.circle(annotated, neck_px, 6, (255,255,255), -1)
                cv2.circle(annotated, neck_px, 6, (0,0,0), 1)

            if sh_ok and hp_ok:
                cv2.line(annotated, neck_px, hip_px, (150,255,150), 3, cv2.LINE_AA)

            for i,j in CONNECTIONS:
                if vis[i]<VIS_THRESHOLD or vis[j]<VIS_THRESHOLD: continue
                c = COLORS.get((i,j),(180,180,180))
                cv2.line(annotated, to_pt(lm[i],w,h), to_pt(lm[j],w,h), c, 3, cv2.LINE_AA)

            for idx in range(33):
                if idx in SKIP_IDS or vis[idx]<VIS_THRESHOLD: continue
                pt = to_pt(lm[idx],w,h)
                cv2.circle(annotated, pt, 6, (255,255,255), -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, 6, (0,0,0), 1, cv2.LINE_AA)

            if hp_ok:
                cv2.circle(annotated, hip_px, 6, (255,255,255), -1)
                cv2.circle(annotated, hip_px, 6, (0,0,0), 1)

            def w3(i): return [wlm[i].x, wlm[i].y, wlm[i].z]
            def neck_w3(): return [(wlm[11].x+wlm[12].x)/2,(wlm[11].y+wlm[12].y)/2,(wlm[11].z+wlm[12].z)/2]

            angles = {}
            if vis[11]>=VIS_THRESHOLD and vis[13]>=VIS_THRESHOLD:
                angles['left_shoulder'] = (calc_angle_3d(neck_w3(),w3(11),w3(13)), to_pt(lm[11],w,h))
            if vis[12]>=VIS_THRESHOLD and vis[14]>=VIS_THRESHOLD:
                angles['right_shoulder'] = (calc_angle_3d(neck_w3(),w3(12),w3(14)), to_pt(lm[12],w,h))
            if all(vis[i]>=VIS_THRESHOLD for i in [11,13,15]):
                angles['left_elbow'] = (calc_angle_3d(w3(11),w3(13),w3(15)), to_pt(lm[13],w,h))
            if all(vis[i]>=VIS_THRESHOLD for i in [12,14,16]):
                angles['right_elbow'] = (calc_angle_3d(w3(12),w3(14),w3(16)), to_pt(lm[14],w,h))
            if all(vis[i]>=VIS_THRESHOLD for i in [11,23,25]):
                angles['left_hip'] = (calc_angle_3d(w3(11),w3(23),w3(25)), to_pt(lm[23],w,h))
            if all(vis[i]>=VIS_THRESHOLD for i in [12,24,26]):
                angles['right_hip'] = (calc_angle_3d(w3(12),w3(24),w3(26)), to_pt(lm[24],w,h))
            if all(vis[i]>=VIS_THRESHOLD for i in [23,25,27]):
                angles['left_knee'] = (calc_angle_3d(w3(23),w3(25),w3(27)), to_pt(lm[25],w,h))
            if all(vis[i]>=VIS_THRESHOLD for i in [24,26,28]):
                angles['right_knee'] = (calc_angle_3d(w3(24),w3(26),w3(28)), to_pt(lm[26],w,h))

            for name, (angle, pt) in angles.items():
                draw_angle(annotated, angle, pt)

        t_now = time.time()
        fps_d = 0.9*fps_d + 0.1/(t_now-t_prev+1e-6)
        t_prev = t_now

        cv2.putText(annotated, f"FPS: {fps_d:.1f}  [Mobile MediaPipe]",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow('Mobile Pose Detection  [Q to quit]', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'): break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()