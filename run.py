import torch
from ultralytics import YOLO
import cv2
import numpy as np
from boxmot import OcSort

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run():
    model = YOLO("yolo11m.pt")
    video_paths = [
        r"AICity22_Track1_MTMC_Tracking\test\S06\c041/vdo.avi",
        r"AICity22_Track1_MTMC_Tracking\test\S06\c042/vdo.avi",
    ]
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]

    trackers = [OcSort() for _ in video_paths]
    window_names = [f"Video {i}" for i in range(len(video_paths))]

    for window_name in window_names:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 360)

    while True:
        rets_and_frames = [cap.read() for cap in caps]
        rets = [ret for ret, _ in rets_and_frames]
        frames = [frame for _, frame in rets_and_frames]

        if not all(rets):
            break

        results = model.predict(
            source=frames,
            verbose=False,
            classes=[2, 3, 5, 7],
            conf=0.25
        )

        for i, (frame, result, tracker) in enumerate(zip(frames, results, trackers)):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy().reshape(-1, 1)
                clss = result.boxes.cls.cpu().numpy().reshape(-1, 1)

                detections = np.hstack((boxes, confs, clss)).astype(np.float32)
                tracks = tracker.update(detections, frame)
            else:
                tracks = np.empty((0, 8), dtype=np.float32)

            for track in tracks:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            display_frame = cv2.resize(frame, (640, 360))
            cv2.imshow(window_names[i], display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()