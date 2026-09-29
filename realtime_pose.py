import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

model = YOLO('yolov8n-pose.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ 使用裝置：{device}")
model.to(device)

SLIM_IDS_YOLO = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

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

VIS_THRESHOLD = 0.5

def to_pt(arr):
    return (int(arr[0]), int(arr[1]))

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_display = 0
    t_prev = time.time()

    print("✓ 開始串流，按 Q 離開")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=0.3)
        annotated = frame.copy()

        for r in results:
            if r.keypoints is None:
                continue
            kps = r.keypoints.data
            if len(kps) == 0:
                continue

            kp = kps[0].cpu().numpy()
            pixel = {i: kp[i, :2] for i in SLIM_IDS_YOLO}
            vis   = {i: kp[i, 2]  for i in SLIM_IDS_YOLO}

            neck_px = ((pixel[5] + pixel[6]) / 2).astype(int)
            hip_px  = ((pixel[11] + pixel[12]) / 2).astype(int)
            head_px = pixel[0].astype(int)

            shoulder_ok = vis[5] >= VIS_THRESHOLD and vis[6] >= VIS_THRESHOLD
            hip_ok      = vis[11] >= VIS_THRESHOLD and vis[12] >= VIS_THRESHOLD

            if vis[0] >= VIS_THRESHOLD and shoulder_ok:
                cv2.line(annotated, to_pt(head_px), to_pt(neck_px),
                         (0,215,255), 3, cv2.LINE_AA)

            if shoulder_ok and hip_ok:
                cv2.line(annotated, to_pt(neck_px), to_pt(hip_px),
                         (150,255,150), 3, cv2.LINE_AA)

            for i, j in SLIM_CONNECTIONS_YOLO:
                if vis[i] < VIS_THRESHOLD or vis[j] < VIS_THRESHOLD:
                    continue
                color = SLIM_COLORS_BGR.get((i,j), (180,180,180))
                cv2.line(annotated, to_pt(pixel[i]), to_pt(pixel[j]),
                         color, 3, cv2.LINE_AA)

            for idx in SLIM_IDS_YOLO:
                dot_color = (100,100,100) if vis[idx] < VIS_THRESHOLD \
                            else (255,255,255)
                cv2.circle(annotated, to_pt(pixel[idx]), 6, dot_color, -1, cv2.LINE_AA)
                cv2.circle(annotated, to_pt(pixel[idx]), 6, (0,0,0), 1, cv2.LINE_AA)

            if shoulder_ok:
                cv2.circle(annotated, to_pt(neck_px), 6, (255,255,255), -1)
                cv2.circle(annotated, to_pt(neck_px), 6, (0,0,0), 1)
            if hip_ok:
                cv2.circle(annotated, to_pt(hip_px), 6, (255,255,255), -1)
                cv2.circle(annotated, to_pt(hip_px), 6, (0,0,0), 1)

        t_now = time.time()
        fps_display = 0.9 * fps_display + 0.1 / (t_now - t_prev + 1e-6)
        t_prev = t_now

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