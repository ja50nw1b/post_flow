import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

# ── 模型載入 ─────────────────────────────────────────────
model = YOLO('yolov8l-pose.pt')

# ── 裝置選擇 ─────────────────────────────────────────────
def select_device():
    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        print("✓ CUDA not available, using CPU")
        return 'cpu'

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nAvailable devices:")
    print(f"  [0] CPU")
    print(f"  [1] GPU - {gpu_name} (CUDA)")

    while True:
        try:
            choice = int(input("\nSelect device [0/1]: "))
            if choice == 0:
                print("✓ Using CPU")
                return 'cpu'
            elif choice == 1:
                print(f"✓ Using GPU: {gpu_name}")
                return 'cuda'
            else:
                print("❌ Invalid, please enter 0 or 1")
        except ValueError:
            print("❌ Please enter a number")

device = select_device()
model.to(device)
torch.set_num_threads(16)

# ── 關節定義 ─────────────────────────────────────────────
SLIM_IDS_YOLO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

LEFT_EYE  = 1
RIGHT_EYE = 2
LEFT_EAR  = 3
RIGHT_EAR = 4

SLIM_CONNECTIONS_YOLO = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

SLIM_COLORS_BGR = {
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
SKIP_IDS              = [0, 1, 2, 3, 4]

# ── 工具函式 ─────────────────────────────────────────────
def to_pt(arr):
    return (int(arr[0]), int(arr[1]))

def estimate_head(pixel, vis, neck_px):
    """鼻子 + 眼睛 + 耳朵加權定位，無平滑"""
    nose_ok      = vis[0]         >= VIS_THRESHOLD_DISPLAY
    left_eye_ok  = vis[LEFT_EYE]  >= VIS_THRESHOLD_DISPLAY
    right_eye_ok = vis[RIGHT_EYE] >= VIS_THRESHOLD_DISPLAY
    left_ear_ok  = vis[LEFT_EAR]  >= VIS_THRESHOLD_DISPLAY
    right_ear_ok = vis[RIGHT_EAR] >= VIS_THRESHOLD_DISPLAY

    pts     = []
    weights = []

    if nose_ok:
        pts.append(pixel[0])
        weights.append(0.1)
    if left_eye_ok:
        pts.append(pixel[LEFT_EYE])
        weights.append(0.2)
    if right_eye_ok:
        pts.append(pixel[RIGHT_EYE])
        weights.append(0.2)
    if left_ear_ok:
        pts.append(pixel[LEFT_EAR])
        weights.append(0.35)
    if right_ear_ok:
        pts.append(pixel[RIGHT_EAR])
        weights.append(0.35)

    if pts:
        total_w     = sum(weights)
        head_center = sum(p * w for p, w in zip(pts, weights)) / total_w
        return head_center.astype(int)
    else:
        shoulder_dist = np.linalg.norm(pixel[5] - pixel[6])
        head_offset   = max(shoulder_dist * 0.7, 40)
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

        results   = model(frame, verbose=False, conf=0.3)
        annotated = frame.copy()

        for r in results:
            if r.keypoints is None:
                continue
            kps = r.keypoints.data
            if len(kps) == 0:
                continue

            kp    = kps[0].cpu().numpy()
            pixel = {i: kp[i, :2] for i in SLIM_IDS_YOLO}
            vis   = {i: kp[i, 2]  for i in SLIM_IDS_YOLO}

            neck_px = ((pixel[5] + pixel[6]) / 2).astype(int)
            hip_px  = ((pixel[11] + pixel[12]) / 2).astype(int)

            shoulder_ok = vis[5]  >= VIS_THRESHOLD_DISPLAY and vis[6]  >= VIS_THRESHOLD_DISPLAY
            hip_ok      = vis[11] >= VIS_THRESHOLD_DISPLAY and vis[12] >= VIS_THRESHOLD_DISPLAY

            # ── 頭部位置 ──────────────────────────────────
            if shoulder_ok:
                head_px = estimate_head(pixel, vis, neck_px)

                cv2.line(annotated, to_pt(neck_px), to_pt(head_px),
                         (0,215,255), 3, cv2.LINE_AA)
                cv2.circle(annotated, to_pt(head_px), 18,
                           (200,220,255), -1, cv2.LINE_AA)
                cv2.circle(annotated, to_pt(head_px), 18,
                           (0,0,0), 1, cv2.LINE_AA)
                cv2.circle(annotated, to_pt(neck_px), 6, (255,255,255), -1)
                cv2.circle(annotated, to_pt(neck_px), 6, (0,0,0), 1)

            # ── 軀幹 ──────────────────────────────────────
            if shoulder_ok and hip_ok:
                cv2.line(annotated, to_pt(neck_px), to_pt(hip_px),
                         (150,255,150), 3, cv2.LINE_AA)

            # ── 四肢 ──────────────────────────────────────
            for i, j in SLIM_CONNECTIONS_YOLO:
                if vis[i] < VIS_THRESHOLD_DISPLAY or vis[j] < VIS_THRESHOLD_DISPLAY:
                    continue
                color = SLIM_COLORS_BGR.get((i,j), (180,180,180))
                cv2.line(annotated, to_pt(pixel[i]), to_pt(pixel[j]),
                         color, 3, cv2.LINE_AA)

            # ── 關節球 ────────────────────────────────────
            for idx in SLIM_IDS_YOLO:
                if idx in SKIP_IDS:
                    continue
                dot_color = (100,100,100) if vis[idx] < VIS_THRESHOLD_DISPLAY \
                            else (255,255,255)
                cv2.circle(annotated, to_pt(pixel[idx]), 6, dot_color, -1, cv2.LINE_AA)
                cv2.circle(annotated, to_pt(pixel[idx]), 6, (0,0,0),   1, cv2.LINE_AA)

            if hip_ok:
                cv2.circle(annotated, to_pt(hip_px), 6, (255,255,255), -1)
                cv2.circle(annotated, to_pt(hip_px), 6, (0,0,0), 1)

        # FPS
        t_now       = time.time()
        fps_display = 0.9 * fps_display + 0.1 / (t_now - t_prev + 1e-6)
        t_prev      = t_now

        cv2.putText(annotated, f"FPS: {fps_display:.1f}  [{device.upper()}]",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow('Pose Detection  [Q to quit]', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()