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
POSE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

LEFT_EYE  = 1
RIGHT_EYE = 2
LEFT_EAR  = 3
RIGHT_EAR = 4

POSE_CONNECTIONS = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

POSE_COLORS = {
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

def draw_skeleton(annotated, pixel, vis):
    """畫骨架、關節球、頭球"""
    neck_px = ((pixel[5] + pixel[6]) / 2).astype(int)
    hip_px  = ((pixel[11] + pixel[12]) / 2).astype(int)

    shoulder_ok = vis[5]  >= VIS_THRESHOLD_DISPLAY and vis[6]  >= VIS_THRESHOLD_DISPLAY
    hip_ok      = vis[11] >= VIS_THRESHOLD_DISPLAY and vis[12] >= VIS_THRESHOLD_DISPLAY

    # 頭球
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
    for i, j in POSE_CONNECTIONS:
        if vis[i] < VIS_THRESHOLD_DISPLAY or vis[j] < VIS_THRESHOLD_DISPLAY:
            continue
        color = POSE_COLORS.get((i,j), (180,180,180))
        cv2.line(annotated, to_pt(pixel[i]), to_pt(pixel[j]),
                 color, 3, cv2.LINE_AA)

    # 關節球
    for idx in POSE_IDS:
        if idx in SKIP_IDS:
            continue
        dot_color = (100,100,100) if vis[idx] < VIS_THRESHOLD_DISPLAY \
                    else (255,255,255)
        cv2.circle(annotated, to_pt(pixel[idx]), 6, dot_color, -1, cv2.LINE_AA)
        cv2.circle(annotated, to_pt(pixel[idx]), 6, (0,0,0), 1, cv2.LINE_AA)

    if hip_ok:
        cv2.circle(annotated, to_pt(hip_px), 6, (255,255,255), -1)
        cv2.circle(annotated, to_pt(hip_px), 6, (0,0,0), 1)

    return neck_px, hip_px, shoulder_ok, hip_ok

def detect_and_draw(frame):
    """YOLO 偵測 + 畫骨架，回傳 annotated 和關節點資料"""
    results   = model(frame, verbose=False, conf=0.3)
    annotated = frame.copy()
    pixel     = None
    vis       = None

    for r in results:
        if r.keypoints is None: continue
        kps = r.keypoints.data
        if len(kps) == 0: continue

        kp    = kps[0].cpu().numpy()
        pixel = {i: kp[i, :2] for i in POSE_IDS}
        vis   = {i: kp[i, 2]  for i in POSE_IDS}
        draw_skeleton(annotated, pixel, vis)
        break

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _, _ = detect_and_draw(frame)

        t_now       = time.time()
        fps_display = 0.9 * fps_display + 0.1 / (t_now - t_prev + 1e-6)
        t_prev      = t_now

        cv2.putText(annotated, f"FPS: {fps_display:.1f}  [{device.upper()}]",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow('Pose Detection  [Q to quit]', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    