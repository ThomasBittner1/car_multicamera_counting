import torch
from ultralytics import YOLO
import cv2

def run():
    model = YOLO("yolo11m.pt")
    video_paths = [r"AICity22_Track1_MTMC_Tracking\test\S06\c041/vdo.avi", r"AICity22_Track1_MTMC_Tracking\test\S06\c042/vdo.avi"]
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]

    while True:
        rets_and_frames = [cap.read() for cap in caps]
        rets = [ret for ret, _ in rets_and_frames]
        frames = [frame for _, frame in rets_and_frames]

        if not all(rets):
            break

        for i, frame in enumerate(frames):
            cv2.imshow(f"Video {i}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
